import taichi as ti
from eng.solver_sph_base import SPHBase
from eng.type_define import *


class SPHSolverTEST(SPHBase):

    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("SPH test starts to serve!")

    @ti.func
    def calc_d_vel_task(self, i, j, ret: ti.template()):
        xij = self.ps.pt[i].x - self.ps.pt[j].x
        ret += xij.x**2 - xij.y + ti.sqrt(abs(xij.z))

    @ti.func
    def calc_v(self, pti):
        return type_vec3f(2 * pti.x.x + 3 * pti.x.y, -3 * pti.x.x - pti.x.y, 2 * pti.x.z)
        # return type_vec3f(pti.x.x, pti.x.y, 2 * pti.x.z)
        # return type_vec3f(pti.x.x + pti.x.y, pti.x.y - pti.x.z**2, 2 * pti.x.z)

    @ti.func
    def calc_d_v_grad(self, cur_v_grad):
        return cur_v_grad - type_mat3f([[2, 3, 0], [-3, -1, 0], [0, 0, 2]])

    @ti.kernel
    def one_step(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].v_tmp = self.calc_v(self.ps.pt[i])

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                tmp = type_mat3f(0)
                self.ps.for_all_neighbors(i, self.calc_v_grad_task, tmp)
                print("========== pt neighbour", i, self.ps.pt[i].x[0:2], "==========")
                # self.ps.pt[i].d_vel = 10*tmp + self.g
                self.ps.pt[i].v_grad = self.calc_d_v_grad(tmp)


