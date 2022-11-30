import taichi as ti
from eng.solver_sph_base import SPHBase
from eng.type_define import *


class DPSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Drucker-Prager SPH starts to serve!")

        # ! now only for one kind of soil
        self.density0 = self.ps.mat_soil[0]["density0"]
        self.coh = self.ps.mat_soil[0]["cohesion"]
        self.fric = self.ps.mat_soil[0]["friction"] / 180 * ti.math.pi
        self.E = self.ps.mat_soil[0]["EYoungMod"]
        self.poi = self.ps.mat_soil[0]["poison"]
        self.dila = self.ps.mat_soil[0]["dilatancy"] / 180 * ti.math.pi

        self.vsound2 = self.E / self.density0
        self.vsound = ti.sqrt(self.vsound2)
        self.dt[None] = self.calc_dt_CFL(CFL_component=0.2, vsound=self.vsound, dt_min=self.dt_min)

        # calculated paras
        self.alpha_fric = ti.tan(self.fric) / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.k_c = 3 * self.coh / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.G = self.E / (2 * (1 + self.poi))
        self.K = self.E / (3 * (1 - 2 * self.poi))
        self.De = ti.Matrix([[1.0 - self.poi, self.poi, 0.0, self.poi],
                            [self.poi, 1.0 - self.poi, 0.0, self.poi],
                            [0.0, 0.0, (1.0 - 2.0 * self.poi) * 0.5, 0.0],
                            [self.poi, self.poi, 0.0, 1.0 - self.poi]]) * (self.E / ((1.0 + self.poi) * (1.0 - 2.0 * self.poi)))

        self.init_stress(self.density0, self.fric)


    ##############################################
    # Tasks
    ##############################################
    @ti.func
    def calc_d_vel_task(self, i, j, ret: ti.template()):
        if self.ps.is_dummy_particle(j):
            self.ps.pt[j].stress_tmp = self.ps.pt[i].stress_tmp
            self.ps.pt[j].density_tmp = self.density0
        arti_visco = self.calc_arti_viscosity_task(0.0, 0.0, i, j, self.vsound) if self.ps.is_soil_particle(j) else 0.0
        tmp = self.ps.pt[j].m_V * self.ps.pt[j].density_tmp * (self.ps.pt[j].stress_tmp / self.ps.pt[j].density_tmp**2 + self.ps.pt[i].stress_tmp / self.ps.pt[i].density_tmp**2 + arti_visco * self.I3) @ self.kernel_deriv_corr(i, j)
        ret += tmp
        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel -= tmp


    ##############################################
    # Stress related functions
    ##############################################
    @ti.func
    def cal_stress_s(self, stress):
        res = stress - stress.trace() / 3.0 * self.I3
        return res

    @ti.func
    def cal_I1(self, stress):
        res = stress.trace()
        return res

    @ti.func
    def cal_sJ2(self, s):
        res = ti.sqrt(0.5 * (s * s).sum())
        return res

    @ti.func
    def cal_fDP(self, I1, sJ2):
        res = sJ2 + self.alpha_fric * I1 - self.k_c
        return res

    @ti.func
    def cal_from_stress(self, stress):
        stress_s = self.cal_stress_s(stress)
        vI1 = self.cal_I1(stress)
        sJ2 = self.cal_sJ2(stress_s)
        fDP = self.cal_fDP(vI1, sJ2)
        return stress_s, vI1, sJ2, fDP


    ##############################################
    # Stress adaptation
    ##############################################
    @ti.func
    def adapt_stress(self, stress):
        # TODO: add a return of the new DP flag and adaptation flag
        # TODO: what is the usage of dfDP?
        res = stress
        stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(res)
        if fDP_new > 1e-4:
            if fDP_new > sJ2:
                res = self.adapt_1(res, vI1)
            stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(res)
            res = self.adapt_2(stress_s, vI1, sJ2)
        return res

    @ti.func
    def chk_flag_DP(self, fDP_new, sJ2):
        flag = 0
        if fDP_new > self.epsilon:
            if fDP_new >= sJ2:
                flag = 1
            else:
                flag = 2
        return flag

    @ti.func
    def adapt_1(self, stress, I1):
        tmp = (I1-self.k_c/self.alpha_fric) / 3.0
        res = stress - tmp * self.I3
        return res

    @ti.func
    def adapt_2(self, s, I1, sJ2):
        r = (-I1 * self.alpha_fric + self.k_c) / sJ2
        res = r * s + self.I3 * I1 / 3.0
        return res


    ##############################################
    # One step
    ##############################################
    @ti.func
    def calc_d_stress_Bui2008(self, pti):
        d_stress = type_mat3f(0)
        d_strain_equ = 0.0
        stress_s, vI1, sJ2, fDP_old = self.cal_from_stress(pti.stress_tmp)
        strain_r = 0.5 * (pti.v_grad +pti.v_grad.transpose())
        spin_r = 0.5 * (pti.v_grad - pti.v_grad.transpose())

        tmj = ti.Matrix([[
            pti.stress_tmp[i, 0] * spin_r[j, 0] +
            pti.stress_tmp[i, 1] * spin_r[j, 1] +
            pti.stress_tmp[i, 2] * spin_r[j, 2] +
            pti.stress_tmp[0, j] * spin_r[i, 0] +
            pti.stress_tmp[1, j] * spin_r[i, 1] +
            pti.stress_tmp[2, j] * spin_r[i, 2] for j in range(self.ps.dim3)] for i in range(self.ps.dim3)])

        strain_r_equ = strain_r - strain_r.trace() / 3.0 * self.I3
        tmp_v = 2.0 * self.G * strain_r_equ + self.K * strain_r.trace() * self.I3

        lambda_r = 0.0
        tmp_g = type_mat3f(0)
        if fDP_old >= -self.epsilon and sJ2 > self.epsilon:
            lambda_r = (3.0 * self.alpha_fric * self.K * strain_r.trace() + (self.G / sJ2) * (stress_s * strain_r).sum()) / (
                        27.0 * self.alpha_fric * self.K * ti.sin(self.dila) + self.G)
            tmp_g = lambda_r * (9.0 * self.K * ti.sin(self.dila) * self.I3 + self.G / sJ2 * stress_s)

        d_stress = tmj + tmp_v - tmp_g
        d_strain_equ = ti.sqrt((strain_r_equ * strain_r_equ).sum() * 2 / 3)

        return d_stress, d_strain_equ

    @ti.kernel
    def one_step(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_soil_particle(i):
                # vel gradient
                v_grad = type_mat3f(0)
                self.ps.for_all_neighbors(i, self.calc_v_grad_task, v_grad)
                self.ps.pt[i].v_grad = v_grad

                # d density
                dd = 0.0
                self.ps.for_all_neighbors(i, self.calc_d_density_task, dd)
                self.ps.pt[i].d_density = dd * self.ps.pt[i].density_tmp

                # d stress
                self.ps.pt[i].d_stress, self.ps.pt[i].d_strain_equ = self.calc_d_stress_Bui2008(self.ps.pt[i])

                # d vel
                d_v = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_d_vel_task, d_v)
                self.ps.pt[i].d_vel = d_v + self.g


    @ti.func
    def advect_something_func(self, i):
        if self.ps.is_soil_particle(i):
            self.chk_density(i, self.density0)

            # adapt stress
            self.adapt_stress(self.ps.pt[i].stress)

            # advect strain equ
            self.ps.pt[i].strain_equ += self.dt[None] * self.ps.pt[i].d_strain_equ
