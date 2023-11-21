import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP, K_SPACE, K_n, K_TAB, K_m, K_RIGHT, K_LEFT, K_UP, K_DOWN, K_EQUALS, K_MINUS
import math
import random
import numpy as np
import pandas as pd
import pickle
from projectile_nn import train_neural_net, nn_predict
import projectile_nn
import torch
import os

# check if running in google colab
try:
    import google.colab
    from google.colab import drive
    # mount google drive
    drive.mount('/content/drive')
    # Set the working directory
    os.chdir('/content/drive/MyDrive/Projects/Python/projectile_motion')
    # make device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running in Google Colab with device: {device}')
except ModuleNotFoundError:
    device = torch.device("mps")
    print(f'Running Locally with device: {device}')

class Ball:
  def __init__(self, x_start, y_start, vel_start, angle, radius):
    self.x_start = x_start
    self.x = self.x_start
    self.y_start = y_start
    self.y = self.y_start
    self.angle = angle
    self.vel_start = vel_start
    self.vx = self.vel_start * math.cos(math.radians(self.angle))
    self.vy = self.vel_start * math.sin(math.radians(self.angle))
    self.gravity = .1
    self.color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)) if train_mode else (0, 0, 0)
    self.radius = radius
    self.fired = False
    self.has_landed = False
    self.distance = None

  def update(self, is_fired):
    if is_fired and not self.has_landed:
      self.x = self.x + self.vx
      if self.y - self.vy < self.y_start:
        self.y = self.y - self.vy
      else:
        self.y = self.y_start + random.randint(1, 5)
      self.vy -= self.gravity
      if self.y > self.y_start:
        self.landed()
    if not is_fired:
      self.vx = self.vel_start * math.cos(math.radians(self.angle))
      self.vy = self.vel_start * math.sin(math.radians(self.angle))

  def landed(self):
    global results
    self.fired = False
    self.has_landed = True
    self.distance = self.x - self.x_start
    # print(f'Start Vel: {self.vel_start} m/s at {self.angle} degrees = {self.distance}')
    data_point = {'vel_start': self.vel_start, 'angle': self.angle, 'distance': self.distance}
    results.append(data_point)

pygame.init()

# number of linear layers
linear_layers = 3
training_epochs = 1000

# Set up font
font = pygame.font.Font(None, 24)

# create list to hold final values
results = []

# Set up display
width, height = 400, 300
width, height = 800, 600
DISPLAYSURF = pygame.display.set_mode((width, height))
pygame.display.set_caption('Train a Neural Net to Launch a Projectile')

# Colors
sky_blue = (120, 180, 230)  # RGB values for sky blue
green = (0, 150, 0)  # RGB values for green

# Set up green rectangle
rect_height = int(0.1 * height)
green_rect = pygame.Rect(0, height - rect_height, width, rect_height)

# ball start position
x_start = 10
y_start = height - rect_height
radius = 5

# bool to control ball motion
is_fired = False

# set mode
train_mode = True

# create to store balls
ball_list = []

# create instance of ball max_vel=4
# for vel in np.arange(1, 4, 0.1):
#   for angle in np.arange(1, 90, 1):
#     ball = Ball(x_start=x_start, y_start=y_start, vel_start=vel, angle=angle, radius=radius)
#     ball_list.append(ball)

ball_count = 10000

vel = np.random.uniform(0, 50)
angle = np.random.uniform(0, 90)
ball = Ball(x_start=x_start, y_start=y_start, vel_start=vel, angle=angle, radius=radius)
ball_list.append(ball)

# for _ in range(ball_count):
#   vel = np.random.uniform(0, 50)
#   angle = np.random.uniform(0, 90)
#   ball = Ball(x_start=x_start, y_start=y_start, vel_start=vel, angle=angle, radius=radius)
#   ball_list.append(ball)
  

# Load target png
target_image = pygame.image.load('media/target.png')
target_image = pygame.transform.scale(target_image, (int(target_image.get_width() * 0.05), int(target_image.get_height() * 0.05)))
target_rect = target_image.get_rect()

# set initial target distance
target_distance = width / 2

# set initial aim angle
aim_angle = 45

# set initial velocity
launch_velocity = 2.0

clock = pygame.time.Clock()

# var for smooth control
target_change = 0
angle_change = 0
vel_change = 0

# draw arrow during test
def draw_arrow(launch_angle):
  length = 5 * launch_velocity + 5
  end_x = x_start + length * math.cos(math.radians(launch_angle))
  end_y = y_start - length * math.sin(math.radians(launch_angle))

  pygame.draw.line(DISPLAYSURF, (255, 0, 0), (x_start, y_start), (end_x, end_y), 2)

while True:
  for event in pygame.event.get():
    if event.type == QUIT:
      pygame.quit()
      sys.exit()
    elif event.type == KEYDOWN:
        if event.key == K_SPACE:
            if not is_fired:
                if train_mode:
                    for _ in range(ball_count - 1):
                        vel = np.random.uniform(0, 50)
                        angle = np.random.uniform(0, 90)
                        ball = Ball(x_start=x_start, y_start=y_start, vel_start=vel, angle=angle, radius=radius)
                        ball_list.append(ball)
                is_fired = True
            else:
              if not train_mode:
                # launch_velocity = nn_predict(nn_model, aim_angle, target_distance, df, device)
                ball_list = []
        # elif event.key == K_n and not train_mode:
        #     launch_velocity = nn_predict(nn_model, aim_angle, target_distance, df, device)
        elif event.key == K_RIGHT:
          if train_mode and linear_layers < 10:
            linear_layers += 1
          else:
            target_change = 1
        elif event.key == K_LEFT:
          if train_mode and linear_layers > 3:
            linear_layers -= 1
          else:
            target_change = -1
        elif event.key == K_UP:
          if train_mode and training_epochs < 100000:
            training_epochs += 1000
          else:
            angle_change = 1
        elif event.key == K_DOWN:
          if train_mode and training_epochs > 1000:
            training_epochs -= 1000
          else:
            angle_change = -1
        elif event.key == K_EQUALS:
          if train_mode and ball_count < 100000:
            ball_count += 1000
          else:
            vel_change = 0.1
        elif event.key == K_MINUS:
          if train_mode and ball_count > 1000:
            ball_count -= 1000
          else:
            vel_change = -0.1
    elif event.type == KEYUP:
      if event.key == K_RIGHT or event.key == K_LEFT:
        target_change = 0
        if not train_mode:
          launch_velocity = nn_predict(nn_model, aim_angle, target_distance, df, device)
      elif event.key == K_UP or event.key == K_DOWN:
        angle_change = 0
        if not train_mode:
          launch_velocity = nn_predict(nn_model, aim_angle, target_distance, df, device)

  # update aim angle
  aim_angle += angle_change
  if aim_angle < 1 or 89 < aim_angle:
    aim_angle -= angle_change
    
  # move target
  target_distance += target_change
  if target_distance < 40 or width < target_distance:
    target_distance -= target_change

  # update launch velocity
  launch_velocity += vel_change
  if launch_velocity < 0.1 or 15 < launch_velocity:
    launch_velocity -= vel_change
        
  # Draw background
  DISPLAYSURF.fill(sky_blue)

  # Draw green rectangle
  pygame.draw.rect(DISPLAYSURF, green, green_rect)

  # train mode ball launching
  if train_mode:
      # text to display
      text = f"Linear Layers: {linear_layers}          Epochs: {training_epochs:,}          Ball Count: {ball_count:,}"
  else:
    text = f"Test Mode      Target Dist: {round(target_distance)}      Launch Angle: {round(aim_angle)}      Launch Vel: {round(launch_velocity, 1)}"
    # Set the initial position of the target
    target_x = x_start + target_distance - target_rect.width / 2
    target_y = y_start - target_rect.height / 2
    # Draw target image at the specified position
    DISPLAYSURF.blit(target_image, (target_x, target_y))
    draw_arrow(aim_angle)

    # create a test ball if the list is empty
    if len(ball_list) == 0:
      ball = Ball(x_start=x_start, y_start=y_start, vel_start=launch_velocity, angle=aim_angle, radius=radius)
      ball_list.append(ball)

  # track balls in air
  airborne_count = ball_count

  # Draw white ball
  for ball in ball_list:
    if ball.has_landed:
      airborne_count -= 1
    print(f"Airborn count: {airborne_count}")
    # draw the circle
    pygame.draw.circle(DISPLAYSURF, ball.color, (ball.x, ball.y), ball.radius)
    # update ball location
    ball.update(is_fired)
    if not train_mode:
        # update aim angle
        ball.angle = aim_angle
        # update launch velocity
        ball.vel_start = launch_velocity
        if ball.has_landed:
            distance = ball.distance
            # landed_text = f'Distance: {round(ball.distance, 1)}'
            # put text on screen
            # text_surface = font.render(landed_text, True, (255, 255, 255))
            # text_rect = text_surface.get_rect(center=(width // 2, height - 25))
            # DISPLAYSURF.blit(text_surface, text_rect)
    else:
      if airborne_count == 0:
        print('Training neural net')
        nn_model, df = train_neural_net(results, device, linear_layers, training_epochs)
        print(f'Training complete. Results saved with {len(results)} entries.', flush=True)
        with open('results.pkl', 'wb') as file:
            pickle.dump(results, file)
        is_fired = False
        # remove train balls
        ball_list = []
        train_mode = False
    
  # put text on screen
  text_surface = font.render(text, True, (255, 255, 255))
  text_rect = text_surface.get_rect(center=(width // 2, height - 25))
  DISPLAYSURF.blit(text_surface, text_rect)

  pygame.display.update()
  clock.tick(120)  # Limit the frame rate to 60 frames per second