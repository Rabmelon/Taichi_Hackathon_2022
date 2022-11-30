import taichi as ti
from eng.solver_sph_base import SPHBase
from eng.type_define import *


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("WCSPH starts to serve!")

        # ! now only for one kind of fluid
        self.density0 = self.ps.mat_fluid[0]["density0"]
        self.viscosity = self.ps.mat_fluid[0]["viscosity"]
        self.stiffness = self.ps.mat_fluid[0]["stiffness"]
        self.exponent = self.ps.mat_fluid[0]["exponent"]



    ##############################################
    # Tasks
    ##############################################
    @ti.func
    def calc_density_task(self, i, j, ret: ti.template()):
        ret += self.ps.pt[j].mass * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def viscosity_force(self, i, j):
        r = self.ps.pt[i].x - self.ps.pt[j].x
        if self.ps.is_dummy_particle(j) or self.ps.is_rigid(j):
            self.calc_dummy_v_tmp(i, j)
        res = 2 * (self.ps.dim + 2) * self.viscosity * self.ps.pt[j].m_V * (self.ps.pt[i].v_tmp - self.ps.pt[j].v_tmp).dot(r) / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_deriv_corr(i, j)
        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel += -res
        return res

    @ti.func
    def calc_non_pressure_force_task(self, i, j, ret: ti.template()):
        ret += self.viscosity_force(i, j)

    @ti.func
    def pressure_force(self, i, j):
        if self.ps.is_dummy_particle(j) or self.ps.is_rigid(j):
            self.ps.pt[j].density_tmp = self.density0
        res = -self.ps.pt[j].density_tmp * self.ps.pt[j].m_V * (self.ps.pt[i].pressure / self.ps.pt[i].density_tmp**2 + self.ps.pt[j].pressure / self.ps.pt[j].density_tmp**2) * self.kernel_deriv_corr(i, j)
        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel += -res
        return res

    @ti.func
    def calc_pressure_force_task(self, i, j, ret: ti.template()):
        ret += self.pressure_force(i, j)


    ##############################################
    # One step
    ##############################################
    @ti.kernel
    def one_step(self):
        # for i in range(self.ps.particle_num[None]):
        #     if self.ps.is_fluid_particle(i):
        #         # density
        #         tmp_density = 0.0
        #         self.ps.for_all_neighbors(i, self.calc_density_task, tmp_density)
        #         # self.ps.pt[i].density = tmp_density
        #         self.ps.pt[i].density_tmp = tmp_density * self.ps.pt[i].CSPM_f

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_dynamic(i):
                self.ps.pt[i].d_vel = type_vec3f(0)

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_fluid_particle(i):
                # d density
                dd = 0.0
                self.ps.for_all_neighbors(i, self.calc_d_density_task, dd)
                self.ps.pt[i].d_density = dd * self.ps.pt[i].density_tmp

                # d vel
                tmp_d_vel = type_vec3f(0)
                self.ps.pt[i].pressure = ti.max(self.stiffness * (ti.pow(self.ps.pt[i].density_tmp / self.density0, self.exponent) - 1.0), 0.0)

                # non-pressure force
                self.ps.for_all_neighbors(i, self.calc_non_pressure_force_task, tmp_d_vel)

                # pressure force
                self.ps.for_all_neighbors(i, self.calc_pressure_force_task, tmp_d_vel)

                self.ps.pt[i].d_vel = tmp_d_vel + self.g

            if self.ps.is_rigid_dynamic(i):
                self.ps.pt[i].d_vel += self.g

            if self.ps.is_rigid_static(i):
                self.ps.pt[i].d_vel = type_vec3f(0)



    @ti.func
    def advect_something_func(self, i):
        if self.ps.is_fluid_particle(i):
            self.chk_density(i, self.density0)

