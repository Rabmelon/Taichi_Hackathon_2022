import taichi as ti
from eng.solver_sph_base import SPHBase
from eng.type_define import *


class MUISPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("μ(I) SPH starts to serve!")

        # ! now only for one kind of soil
        self.density0 = self.ps.mat_soil[0]["density0"]
        self.coh = self.ps.mat_soil[0]["cohesion"]
        self.fric = self.ps.mat_soil[0]["friction"] / 180 * ti.math.pi
        self.E = self.ps.mat_soil[0]["EYoungMod"]
        self.poi = self.ps.mat_soil[0]["poison"]
        self.dila = self.ps.mat_soil[0]["dilatancy"] / 180 * ti.math.pi

        self.eta_0 = 0.0
        self.mu = ti.tan(self.fric)
        self.vsound2 = self.E / self.density0
        self.vsound = ti.sqrt(self.vsound2)
        self.dt[None] = self.calc_dt_CFL(CFL_component=0.2, vsound=self.vsound, dt_min=self.dt_min)


    ##############################################
    # Tasks
    ##############################################
    @ti.func
    def calc_d_vel_task(self, i, j, ret: ti.template()):
        if self.ps.is_dummy_particle(j) or self.ps.is_rigid(j):
            self.ps.pt[j].stress = self.ps.pt[i].stress
            self.ps.pt[j].density_tmp = self.density0
        arti_visco = self.calc_arti_viscosity_task(0.0, 0.0, i, j, self.vsound) if self.ps.is_soil_particle(j) else 0.0
        tmp = self.ps.pt[j].m_V * self.ps.pt[j].density_tmp * (self.ps.pt[j].stress / self.ps.pt[j].density_tmp**2 + self.ps.pt[i].stress / self.ps.pt[i].density_tmp**2 + arti_visco * self.I3) @ self.kernel_deriv_corr(i, j)
        ret += tmp
        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel -= tmp


    ##############################################
    # One step
    ##############################################
    @ti.kernel
    def one_step(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_dynamic(i):
                self.ps.pt[i].d_vel = self.g

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

                # pressure
                self.ps.pt[i].pressure = ti.max(self.vsound2 * (self.ps.pt[i].density - self.density0), 0.0)

                # strain
                strain_r = 0.5 * (self.ps.pt[i].v_grad + self.ps.pt[i].v_grad.transpose())
                strain_r_dbdot = ti.sqrt(0.5 * (strain_r * strain_r).sum()) + self.epsilon

                # stress
                tau = (self.eta_0 + (self.coh + self.ps.pt[i].pressure * self.mu) / strain_r_dbdot) * strain_r
                self.ps.pt[i].stress = tau - self.ps.pt[i].pressure * self.I3

                # d vel
                d_v = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_d_vel_task, d_v)
                self.ps.pt[i].d_vel = d_v + self.g

                # d strain equ
                strain_r_equ = strain_r - strain_r.trace() / 3.0 * self.I3
                self.ps.pt[i].d_strain_equ = ti.sqrt((strain_r_equ * strain_r_equ).sum() * 2 / 3)

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_static(i):
                self.ps.pt[i].d_vel = type_vec3f(0)

    @ti.func
    def advect_something_func(self, i):
        if self.ps.is_soil_particle(i):
            self.chk_density(i, self.density0)
            # advect strain equ
            self.ps.pt[i].strain_equ += self.dt[None] * self.ps.pt[i].d_strain_equ


