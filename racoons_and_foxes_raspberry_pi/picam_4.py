
import pygame
import pygame.camera
from pygame.locals import *

pygame.init()
pygame.camera.init()
size = (640,480)
display = pygame.display.set_mode(size, 0)
clist = pygame.camera.list_cameras()
#print(clist)
if not clist:
    print("No camera detected. Aborting.")
    exit()
cam = pygame.camera.Camera(clist[0], size)
#print(cam)
cam.start()
snapshot = pygame.surface.Surface(size, 0, display)

#cap = Capture()
while True:
    events = pygame.event.get()
    for e in events:
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            # close the camera safely
            cam.stop()
            exit()
    #self.get_and_flip()
    if cam.query_image():
        snapshot = cam.get_image(snapshot)

    # blit it to the display surface.  simple!
    display.blit(snapshot, (0,0))
    pygame.display.flip()

