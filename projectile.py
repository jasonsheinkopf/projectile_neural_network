import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP, K_SPACE, K_RIGHT, K_LEFT, K_UP, K_DOWN, K_EQUALS, K_MINUS, K_r, K_LEFTBRACKET, K_RIGHTBRACKET
import math
import random
import numpy as np
# import pandas as pd
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
    radius = 3

    def __init__(self, x_start, y_start, vel_start, angle):
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
        self.has_landed = False
        self.distance = None

    def update(self, is_fired):
        if is_fired and not self.has_landed:
            # update x location from velocity
            self.x = self.x + self.vx
            # apply gravity
            self.vy -= self.gravity
            # check if next timestep will still be in the air
            if self.y - self.vy < self.y_start:
                # update y location from velocity
                self.y = self.y - self.vy
            # if next time step will be underground
            else:
                # ground at random depth
                self.y = self.y_start + random.randint(1, 5)
                self.landed()
        if not is_fired:
            # update user selected start velocity
            self.vx = self.vel_start * math.cos(math.radians(self.angle))
            self.vy = self.vel_start * math.sin(math.radians(self.angle))

    def landed(self):
        global results
        global is_fired
        global has_landed
        self.has_landed = True
        if not train_mode:
            is_fired = False
            has_landed = True
        self.distance = self.x - self.x_start
        data_point = {'vel_start': self.vel_start, 'angle': self.angle, 'distance': self.distance}
        # append data dict to results list
        results.append(data_point)


def place_text(DISPLAYSURF, text, position):
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=position)
    DISPLAYSURF.blit(text_surface, text_rect)


pygame.init()

# neural network hyperparameters
linear_layers = 10
training_epochs = 20000
ball_count = 25000
neurons = 30
ball_display_count = ball_count

# Set up font
font = pygame.font.Font(None, 24)

# create list to hold final values
results = []

# Set up display
width, height = 400, 300
width, height = 800, 600
DISPLAYSURF = pygame.display.set_mode((width, height))
pygame.display.set_caption('AimNet: Train a Neural Net to Aim a Projectile')

# Colors
sky_blue = (120, 180, 230)
green = (0, 150, 0)

# grass rectangle
rect_height = int(0.1 * height)
green_rect = pygame.Rect(0, height - rect_height, width, rect_height)

# ball start position
x_start = 20
y_start = height - rect_height

# bool to control ball motion
is_fired = False
has_landed = False

# set mode
train_mode = True

# create to store balls
ball_list = []

# create one white ball to show at start
ball = Ball(x_start=x_start, y_start=y_start, vel_start=8, angle=45)
ball.color = (255, 255, 255)
ball_list.append(ball)

# load logo png and scale
logo_scale = 0.1
logo_image = pygame.image.load('media/AimNet.png')
logo_image = pygame.transform.scale(logo_image, (int(logo_image.get_width() * logo_scale), int(logo_image.get_height() * logo_scale)))
logo_rect = logo_image.get_rect()

# load target png and scale
scale = 0.05
target_image = pygame.image.load('media/target.png')
target_image = pygame.transform.scale(target_image, (int(target_image.get_width() * scale), int(target_image.get_height() * scale)))
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


# draw arrow during test
def draw_arrow(launch_angle):
    length = 9 * launch_velocity + 5
    end_x = x_start + length * math.cos(math.radians(launch_angle))
    end_y = y_start - length * math.sin(math.radians(launch_angle))
    pygame.draw.line(DISPLAYSURF, (255, 0, 0), (x_start, y_start), (end_x, end_y), 2)


# main game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                if train_mode and not is_fired:
                    # create random balls and append to list
                    for _ in range(ball_count - 1):
                        angle = np.random.uniform(1, 89)
                        vel_max = math.sqrt(38 / (math.sin(math.radians(angle)) * math.cos(math.radians(angle))))
                        vel = np.random.uniform(1.5, vel_max)
                        ball = Ball(x_start=x_start, y_start=y_start, vel_start=vel, angle=angle)
                        ball_list.append(ball)
                        highest_ball = ball_list[0]
                    is_fired = True
                else:
                    if not train_mode:
                        # delete all training balls
                        ball_list = []
                        is_fired = True
            elif event.key == K_r:
                # bool to control ball motion
                is_fired = False
                has_landed = False

                # set mode
                train_mode = True

                # create to store balls
                ball_list = []

                # create one white ball to show at start
                ball = Ball(x_start=x_start, y_start=y_start, vel_start=8, angle=45)
                ball.color = (255, 255, 255)
                ball_list.append(ball)
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
            elif event.key == K_RIGHTBRACKET and train_mode and neurons < 100:
                neurons += 1
            elif event.key == K_LEFTBRACKET and train_mode and neurons > 6:
                neurons -= 1
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
            if event.key in (K_RIGHT, K_LEFT):
                target_change = 0
            elif event.key in (K_UP, K_DOWN):
                angle_change = 0

    if not is_fired and has_landed:
        # update aim angle
        aim_angle += angle_change
        if aim_angle < 1 or 89 < aim_angle:
            aim_angle -= angle_change

        # move target
        target_distance += target_change
        if target_distance < 40 or width - 40 < target_distance:
            target_distance -= target_change

    # inference neural net to get predicted launch velocity
    if not train_mode:
        launch_velocity = nn_predict(nn_model, aim_angle, target_distance, df, device)

    # Draw sky
    DISPLAYSURF.fill(sky_blue)

    # Draw grass
    pygame.draw.rect(DISPLAYSURF, green, green_rect)

    if train_mode:
        # text to display
        instruction_text = "Press SPACE to launch and train AimNet!"

        left_text = f'Linear Layers (LT / RT)'
        middle_text = f'Neurons ([ / ])'
        right_text = f'Epochs (UP / DN)'
        last_text = f'Ball Count (+ / -)'

        left_value = f'{linear_layers}'
        middle_value = f'{neurons}'
        right_value = f'{training_epochs:,}'
        last_value = f'{ball_display_count:,}'

    else:
        instruction_text = "Choose angle and distance or (r)etrain. AimNet will set launch velocity!"

        left_text = f'Target Dist (LT / RT)'
        middle_text = f'Launch Angle (UP / DN)'
        right_text = f'Launch Velocity'
        last_text = ''

        left_value = f'{round(target_distance)} m'
        middle_value = f'{round(aim_angle)} deg'
        right_value = f'{round(launch_velocity, 1)} m/s'
        last_value = ''

        target_x = x_start + target_distance - target_rect.width / 2
        target_y = y_start - target_rect.height / 2
        # Draw target image at the specified position
        DISPLAYSURF.blit(target_image, (target_x, target_y))
        draw_arrow(aim_angle)

        # create a test ball if the list is empty
        if len(ball_list) == 0:
            ball = Ball(x_start=x_start, y_start=y_start, vel_start=launch_velocity, angle=aim_angle)
            ball_list.append(ball)

    # display logo
    DISPLAYSURF.blit(logo_image, (10, 10))

    # track balls in air
    airborne_count = ball_count

    # Draw balls
    for ball in ball_list:
        if is_fired:
            if ball.y < highest_ball.y:
                highest_ball = ball
        # print(ball.y)
        if ball.has_landed:
            airborne_count -= 1
        # draw the circle
        pygame.draw.circle(DISPLAYSURF, ball.color, (ball.x, ball.y), Ball.radius)
        if ball.has_landed == False:
            # update ball location
            ball.update(is_fired)
        if not train_mode:
            # update aim angle
            ball.angle = aim_angle
            # update launch velocity
            ball.vel_start = launch_velocity
            if ball.has_landed:
                distance = ball.distance
                landed_text = f'Distance: {round(ball.distance, 1)} m'
                error = round(abs(ball.distance - target_distance), 1)
                comparison_text = f'{error} m from target.'

                # instruction text
                place_text(DISPLAYSURF, landed_text, (width // 2, height // 2))
                place_text(DISPLAYSURF, comparison_text, (width // 2, height // 2 + 30))

        # create tracking rectangle when ball is above screen
        if ball.y < 0:
            # set marker width to decrease with height offscreen
            marker_w = 30 + ball.y / 400
            marker = pygame.Rect(ball.x - marker_w / 2, 10, marker_w, 4)
            pygame.draw.rect(DISPLAYSURF, ball.color, marker)
    # detect when all balls have landed
    if airborne_count == 0:
        has_landed = False
        training_text = f'Now training AimNet....'
        place_text(DISPLAYSURF, training_text, (width // 2, height // 2))
        pygame.display.update()
        nn_model, df = train_neural_net(results, device, linear_layers, training_epochs, neurons)
        print(f'Training complete. Results saved with {len(results)} entries.', flush=True)
        with open('results.pkl', 'wb') as file:
            pickle.dump(results, file)
        is_fired = False
        has_landed = True
        # remove train balls
        ball_list = []
        train_mode = False
    
    ball_display_count = airborne_count

    # instruction text
    place_text(DISPLAYSURF, instruction_text, (width // 2, 35))

    # variable name text
    place_text(DISPLAYSURF, left_text, (width // 5, height - 40))
    place_text(DISPLAYSURF, middle_text, (2 * width // 5, height - 40))
    place_text(DISPLAYSURF, right_text, (3 * width // 5, height - 40))
    place_text(DISPLAYSURF, last_text, (4 * width // 5, height - 40))

    # variable value text
    place_text(DISPLAYSURF, left_value, (width // 5, height - 20))
    place_text(DISPLAYSURF, middle_value, (2 * width // 5, height - 20))
    place_text(DISPLAYSURF, right_value, (3 * width // 5, height - 20))
    place_text(DISPLAYSURF, last_value, (4 * width // 5, height - 20))

    pygame.display.update()
    clock.tick(120)