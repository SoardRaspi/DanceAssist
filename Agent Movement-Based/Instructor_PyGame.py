import time
import pygame
import cv2 as cv
import numpy as np

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = int(0.8 * SCREEN_WIDTH)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Instructor")

head = pygame.Rect((375, 135, 50, 50))
body = pygame.Rect((350, 185, 100, 100))
limb1_length = 50
limb2_length = 50
limb3_length = 25

joints = {"rs": (350, 185), "re": (), "rw": (), "r_ft": (), "ls": (450, 185), "le": (), "lw": (), "l_ft": (),
          "rh": (375, 285), "rk": (), "ra": (), "r_tt": (), "lh": (425, 285), "lk": (), "la": (), "l_tt": ()}
angles = {"rs": 0, "re": 0, "rw": 0, "ls": 0, "le": 0, "lw": 0,
          "rh": 0, "rk": 0, "ra": 0, "lh": 0, "lk": 0, "la": 0}

joints_2 = {"rs": (350, 185), "re": (), "rw": (), "r_ft": (), "ls": (450, 185), "le": (), "lw": (), "l_ft": (),
          "rh": (375, 285), "rk": (), "ra": (), "r_tt": (), "lh": (425, 285), "lk": (), "la": (), "l_tt": ()}
angles_2 = {"rs": 0, "re": 0, "rw": 0, "ls": 0, "le": 0, "lw": 0,
          "rh": 0, "rk": 0, "ra": 0, "lh": 0, "lk": 0, "la": 0}


class changer():
    def __init__(self, joints_dir, joints_dir_2, angles_dir, angles_dir_2, l1_l, l2_l, l3_l, l1_l_2, l2_l_2, l3_l_2,
                 screen, head, body):
        self.joints = joints_dir
        self.joints_2 = joints_dir_2

        self.angles = angles_dir
        self.angles_2 = angles_dir_2

        self.l1_l = l1_l
        self.l1_l_2 = l1_l_2

        self.l2_l = l2_l
        self.l2_l_2 = l2_l_2

        self.l3_l = l3_l
        self.l3_l_2 = l3_l_2

        self.head = head
        self.body = body

        self.screen = screen

        self.initialize()
        self.draw()

    def initialize(self):
        self.joints["re"] = (350, 185 + self.l1_l)
        self.joints["rw"] = (350, 185 + self.l1_l + self.l2_l)
        self.joints["r_ft"] = (350, 185 + self.l1_l + self.l2_l + self.l3_l)

        self.joints["le"] = (450, 185 + self.l1_l)
        self.joints["lw"] = (450, 185 + self.l1_l + self.l2_l)
        self.joints["l_ft"] = (450, 185 + self.l1_l + self.l2_l + self.l3_l)

        self.joints["rk"] = (375, 285 + self.l1_l)
        self.joints["ra"] = (375, 285 + self.l1_l + self.l2_l)
        self.joints["r_tt"] = (375, 285 + self.l1_l + self.l2_l + self.l3_l)

        self.joints["lk"] = (425, 285 + self.l1_l)
        self.joints["la"] = (425, 285 + self.l1_l + self.l2_l)
        self.joints["l_tt"] = (425, 285 + self.l1_l + self.l2_l + self.l3_l)

    def draw(self, flag_white=None):
        pygame.draw.line(self.screen, (255, 0, 0), self.joints["rs"], self.joints["re"], 5)
        pygame.draw.line(self.screen, (0, 255, 0), self.joints["re"], self.joints["rw"], 5)
        pygame.draw.line(self.screen, (0, 0, 255), self.joints["rw"], self.joints["r_ft"], 5)

        pygame.draw.line(self.screen, (255, 0, 0), self.joints["ls"], self.joints["le"], 5)
        pygame.draw.line(self.screen, (0, 255, 0), self.joints["le"], self.joints["lw"], 5)
        pygame.draw.line(self.screen, (0, 0, 255), self.joints["lw"], self.joints["l_ft"], 5)

        pygame.draw.line(self.screen, (255, 0, 0), self.joints["rh"], self.joints["rk"], 5)
        pygame.draw.line(self.screen, (0, 255, 0), self.joints["rk"], self.joints["ra"], 5)
        pygame.draw.line(self.screen, (0, 0, 255), self.joints["ra"], self.joints["r_tt"], 5)

        pygame.draw.line(self.screen, (255, 0, 0), self.joints["lh"], self.joints["lk"], 5)
        pygame.draw.line(self.screen, (0, 255, 0), self.joints["lk"], self.joints["la"], 5)
        pygame.draw.line(self.screen, (0, 0, 255), self.joints["la"], self.joints["l_tt"], 5)

        if flag_white:
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["rs"], self.joints_2["re"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["re"], self.joints_2["rw"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["rw"], self.joints_2["r_ft"], 5)

            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["ls"], self.joints_2["le"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["le"], self.joints_2["lw"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["lw"], self.joints_2["l_ft"], 5)

            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["rh"], self.joints_2["rk"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["rk"], self.joints_2["ra"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["ra"], self.joints_2["r_tt"], 5)

            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["lh"], self.joints_2["lk"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["lk"], self.joints_2["la"], 5)
            pygame.draw.line(self.screen, (255, 255, 255), self.joints_2["la"], self.joints_2["l_tt"], 5)

    def __coords_original_arms(self, shoulder, angle_1, l_):
        _angle_1 = angle_1 * np.pi / 180.0

        x_temp = int(np.floor(shoulder[0] - int(l_ * np.sin(_angle_1))))
        y_temp = int(np.floor(shoulder[1] + int(l_ * np.cos(_angle_1))))

        return x_temp, y_temp

    def __coords_original_arms_l2(self, elbow, angle_2, l__):
        _angle_2 = angle_2 * np.pi / 180.0

        x_temp = int(np.floor(elbow[0] - int(l__ * np.sin(_angle_2))))
        y_temp = int(np.floor(elbow[1] + int(l__ * np.cos(_angle_2))))

        return x_temp, y_temp

    def __coords_original_arms_l3(self, wrist, angle_3, l___):
        _angle_3 = angle_3 * np.pi / 180.0

        x_temp = int(np.floor(wrist[0] - int(l___ * np.sin(_angle_3))))
        y_temp = int(np.floor(wrist[1] + int(l___ * np.cos(_angle_3))))

        return x_temp, y_temp

    def change(self, del_rs, del_ls, del_re, del_le, del_rw, del_lw, del_rh, del_lh, del_rk, del_lk, del_ra, del_la,
               del_rs_2, del_ls_2, del_re_2, del_le_2, del_rw_2, del_lw_2, del_rh_2, del_lh_2, del_rk_2, del_lk_2, del_ra_2, del_la_2,
               flag_polygon=None, color_polygon=None):
        if del_rs is None:
            del_rs = 0
        if del_ls is None:
            del_ls = 0
        if del_re is None:
            del_re = 0
        if del_le is None:
            del_le = 0
        if del_rw is None:
            del_rw = 0
        if del_lw is None:
            del_lw = 0
        if del_rh is None:
            del_rh = 0
        if del_lh is None:
            del_lh = 0
        if del_rk is None:
            del_rk = 0
        if del_lk is None:
            del_lk = 0
        if del_ra is None:
            del_ra = 0
        if del_la is None:
            del_la = 0

        self.angles["rs"] = del_rs
        self.angles["ls"] = del_ls
        self.angles["re"] = del_re
        self.angles["le"] = del_le
        self.angles["rw"] = del_rw
        self.angles["lw"] = del_lw

        self.angles["rh"] = del_rh
        self.angles["lh"] = del_lh
        self.angles["rk"] = del_rk
        self.angles["lk"] = del_lk
        self.angles["ra"] = del_ra
        self.angles["la"] = del_la

        new_re = self.__coords_original_arms(self.joints["rs"], self.angles["rs"], self.l1_l)
        self.joints["re"] = new_re
        new_le = self.__coords_original_arms(self.joints["ls"], self.angles["ls"], self.l1_l)
        self.joints["le"] = new_le
        new_rk = self.__coords_original_arms(self.joints["rh"], self.angles["rh"], self.l1_l)
        self.joints["rk"] = new_rk
        new_lk = self.__coords_original_arms(self.joints["lh"], self.angles["lh"], self.l1_l)
        self.joints["lk"] = new_lk

        # __________

        new_rw = self.__coords_original_arms_l2(self.joints["re"], self.angles["re"], self.l2_l)
        self.joints["rw"] = new_rw
        new_lw = self.__coords_original_arms_l2(self.joints["le"], self.angles["le"], self.l2_l)
        self.joints["lw"] = new_lw
        new_ra = self.__coords_original_arms_l2(self.joints["rk"], self.angles["rk"], self.l2_l)
        self.joints["ra"] = new_ra
        new_la = self.__coords_original_arms_l2(self.joints["lk"], self.angles["lk"], self.l2_l)
        self.joints["la"] = new_la

        # __________

        new_r_ft = self.__coords_original_arms_l3(self.joints["rw"], self.angles["rw"], self.l3_l)
        self.joints["r_ft"] = new_r_ft
        new_l_ft = self.__coords_original_arms_l3(self.joints["lw"], self.angles["lw"], self.l3_l)
        self.joints["l_ft"] = new_l_ft
        new_r_tt = self.__coords_original_arms_l3(self.joints["ra"], self.angles["ra"], self.l3_l)
        self.joints["r_tt"] = new_r_tt
        new_l_tt = self.__coords_original_arms_l3(self.joints["la"], self.angles["la"], self.l3_l)
        self.joints["l_tt"] = new_l_tt

        # ___________
        # ___________
        # ___________

        if del_rs_2 is None:
            del_rs_2 = 0
        if del_ls_2 is None:
            del_ls_2 = 0
        if del_re_2 is None:
            del_re_2 = 0
        if del_le_2 is None:
            del_le_2 = 0
        if del_rw_2 is None:
            del_rw_2 = 0
        if del_lw_2 is None:
            del_lw_2 = 0
        if del_rh_2 is None:
            del_rh_2 = 0
        if del_lh_2 is None:
            del_lh_2 = 0
        if del_rk_2 is None:
            del_rk_2 = 0
        if del_lk_2 is None:
            del_lk_2 = 0
        if del_ra_2 is None:
            del_ra_2 = 0
        if del_la_2 is None:
            del_la_2 = 0

        self.angles_2["rs"] = del_rs_2
        self.angles_2["ls"] = del_ls_2
        self.angles_2["re"] = del_re_2
        self.angles_2["le"] = del_le_2
        self.angles_2["rw"] = del_rw_2
        self.angles_2["lw"] = del_lw_2

        self.angles_2["rh"] = del_rh_2
        self.angles_2["lh"] = del_lh_2
        self.angles_2["rk"] = del_rk_2
        self.angles_2["lk"] = del_lk_2
        self.angles_2["ra"] = del_ra_2
        self.angles_2["la"] = del_la_2

        new_re_2 = self.__coords_original_arms(self.joints_2["rs"], self.angles_2["rs"], self.l1_l_2)
        self.joints_2["re"] = new_re_2
        new_le_2 = self.__coords_original_arms(self.joints_2["ls"], self.angles_2["ls"], self.l1_l_2)
        self.joints_2["le"] = new_le_2
        new_rk_2 = self.__coords_original_arms(self.joints_2["rh"], self.angles_2["rh"], self.l1_l_2)
        self.joints_2["rk"] = new_rk_2
        new_lk_2 = self.__coords_original_arms(self.joints_2["lh"], self.angles_2["lh"], self.l1_l_2)
        self.joints_2["lk"] = new_lk_2

        # __________

        new_rw_2 = self.__coords_original_arms_l2(self.joints_2["re"], self.angles_2["re"], self.l2_l_2)
        self.joints_2["rw"] = new_rw_2
        new_lw_2 = self.__coords_original_arms_l2(self.joints_2["le"], self.angles_2["le"], self.l2_l_2)
        self.joints_2["lw"] = new_lw_2
        new_ra_2 = self.__coords_original_arms_l2(self.joints_2["rk"], self.angles_2["rk"], self.l2_l_2)
        self.joints_2["ra"] = new_ra_2
        new_la_2 = self.__coords_original_arms_l2(self.joints_2["lk"], self.angles_2["lk"], self.l2_l_2)
        self.joints_2["la"] = new_la_2

        # __________

        new_r_ft_2 = self.__coords_original_arms_l3(self.joints_2["rw"], self.angles_2["rw"], self.l3_l_2)
        self.joints_2["r_ft"] = new_r_ft_2
        new_l_ft_2 = self.__coords_original_arms_l3(self.joints_2["lw"], self.angles_2["lw"], self.l3_l_2)
        self.joints_2["l_ft"] = new_l_ft_2
        new_r_tt_2 = self.__coords_original_arms_l3(self.joints_2["ra"], self.angles_2["ra"], self.l3_l_2)
        self.joints_2["r_tt"] = new_r_tt_2
        new_l_tt_2 = self.__coords_original_arms_l3(self.joints_2["la"], self.angles_2["la"], self.l3_l_2)
        self.joints_2["l_tt"] = new_l_tt_2

        # __________
        # __________
        # __________

        self.screen.fill((0, 0, 0))

        if (flag_polygon is not None) and flag_polygon:
            points = [self.joints["rs"], self.joints["re"], self.joints["rw"],
                      self.joints_2["rs"], self.joints_2["re"], self.joints_2["rw"]]

            pygame.draw.polygon(self.screen, color_polygon, points, 0)

        pygame.draw.rect(self.screen, (255, 255, 255), self.head, 3)
        pygame.draw.rect(self.screen, (255, 255, 255), self.body, 3)
        self.draw(flag_white=True)

    def get_curr_angles_wrt_vl(self):
        return self.angles


def main_function(changer_obj, screen, head, body):
    run = True
    while run:
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), head, 3)
        pygame.draw.rect(screen, (255, 255, 255), body, 3)

        changer_obj.draw()
        arr_angles = list(map(int, input().split()))
        changer_obj.change(arr_angles[0], arr_angles[1], arr_angles[2], arr_angles[3], arr_angles[4], arr_angles[5],
                           arr_angles[6], arr_angles[7], arr_angles[8], arr_angles[9], arr_angles[10], arr_angles[11])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.update()

    pygame.quit()
