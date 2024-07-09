#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import dearpygui.dearpygui as dpg
import threading

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_1
    from pygame.locals import K_2
    from pygame.locals import K_3
    from pygame.locals import K_4
    from pygame.locals import K_5
    from pygame.locals import K_6
    from pygame.locals import K_7
    from pygame.locals import K_8    
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_j
    from pygame.locals import K_k
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_y
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- Quaternion Class and functions --------------------------------------------
# ==============================================================================
from math import sin, cos, radians
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other):
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def inverse(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

def to_quaternion(rotation):
    """
    Convert a carla.Rotation to a quaternion.
    """
    pitch, yaw, roll = radians(rotation.pitch), radians(rotation.yaw), radians(rotation.roll)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return Quaternion(w, x, y, z)

def lerp(start, end, alpha):
    return start + alpha * (end - start)

def rotate_vector(vector, rotation):
    """
    Rotate a vector by a given rotation.

    Parameters:
    - vector: carla.Vector3D. The vector you want to rotate.
    - rotation: carla.Rotation. The rotation you want to apply.

    Returns:
    - carla.Vector3D. The rotated vector.
    """
    # Convert the given rotation to a quaternion for the rotation operation
    quaternion = to_quaternion(rotation)

    # Convert the given vector to a quaternion where w = 0
    vector_quaternion = Quaternion(0, vector.x, vector.y, vector.z)

    # Perform the rotation using quaternion multiplication
    rotated_vector_quaternion = quaternion * vector_quaternion * quaternion.inverse()

    # Convert the result back to a Vector3D and return
    return carla.Vector3D(rotated_vector_quaternion.x, rotated_vector_quaternion.y, rotated_vector_quaternion.z)


# ==============================================================================
# -- GUI ---------------------------------------------------------------------
# ==============================================================================

def on_button_click(sender, app_data, user_data):
    world = user_data
    world.request_restart()

def on_apply_vehicles(sender, app_data, user_data):
    world, vehicle_input_tag = user_data
    vehicle_number = dpg.get_value(vehicle_input_tag)
    world.default_num_vehicles= vehicle_number

def on_spawn_vehicles(sender, app_data, user_data):
    world, vehicle_input_tag = user_data
    number_of_vehicles = dpg.get_value(vehicle_input_tag)
    world.spawn_vehicles(number_of_vehicles)  # Pass the input number to the spawn method

def on_apply_traffic_conditions(sender, app_data, user_data):
    world, traffic_level_tag = user_data
    traffic_level = dpg.get_value(traffic_level_tag)
    world.tm_conditions(traffic_level)

def on_apply_pedestrians(sender, app_data, user_data):
    world, pedestrian_input_tag = user_data
    number_of_pedestrians = dpg.get_value(pedestrian_input_tag)
    world.spawn_pedestrians(number_of_pedestrians)  # Update the number of pedestrians

def on_apply_props(sender, app_data, user_data):
    world, props_input_tag = user_data
    number_of_props = dpg.get_value(props_input_tag)
    world.spawn_props(number_of_props)  # Update the number of props

def toggle_spectator_mode(sender, app_data, user_data):
    world = user_data
    world.is_spectator_mode = not world.is_spectator_mode
    mode_status = "ON" if world.is_spectator_mode else "OFF"
    world.hud.notification(f"Spectator Mode is {mode_status}")

def on_spectator_to_closest_vehicle(sender, app_data, user_data):
    world = user_data
    world.spectator_to_closest_vehicle()

def on_toggle_autopilot(sender, app_data, user_data):
    world = user_data
    world.toggle_autopilot()

def on_toggle_freeze(sender, app_data, user_data):
    world = user_data
    world.freeze = not world.freeze
    if world.freeze:
        world.hud.notification("World Frozen")
        world.pause_vehicles()  # Make sure this method exists and does what it's supposed to
    else:
        world.hud.notification("World Unfrozen")
        world.resume_vehicles()  # Make sure this method exists and does what it's supposed to

def on_switch_to_previous_vehicle(sender, app_data, user_data):
    world = user_data
    world.switch_to_previous_vehicle()

def on_switch_to_next_vehicle(sender, app_data, user_data):
    world = user_data
    world.switch_to_next_vehicle()

def on_start_recording(sender, app_data, user_data):
    world, file_path = user_data
    # Start recording to the specified file path
    world.client.start_recorder(file_path)
    world.recording_enabled = True
    world.hud.notification("Recorder is ON")

def on_stop_recording(sender, app_data, user_data):
    world = user_data
    # Stop the recorder
    world.client.stop_recorder()
    world.recording_enabled = False
    world.hud.notification("Recorder is OFF")

def on_directory_selected(sender, app_data, user_data):
    # 'app_data' contains the selection information, including the selected directory path
    directory_path = next(iter(app_data['selections'].values()), None) if app_data['selections'] else None

    if directory_path:
        # Retrieve the filename from the text input using the tag from user_data
        file_name = dpg.get_value(user_data['file_name_tag'])
        # Concatenate the directory path and the file name
        full_path = os.path.join(directory_path, file_name)
        # Save the full path to a text item or input text for display or further use
        dpg.set_value(user_data['file_path_tag'], full_path)
    else:
        # User did not select a directory
        dpg.set_value(user_data['file_path_tag'], "No directory selected.")

def on_browse(sender, app_data, user_data):
    # Show the file dialog
    dpg.show_item("file_dialog_id")

def setup_file_dialog(user_data):
    # Check if the file dialog already exists and delete it if it does
    if dpg.does_item_exist("file_dialog_id"):
        dpg.delete_item("file_dialog_id")

    # Now, create the file dialog with the user_data
    dpg.add_file_dialog(
        directory_selector=True,
        show=False,
        callback=on_directory_selected,
        user_data=user_data,  # Pass the user_data dictionary here
        tag="file_dialog_id",
        height=400
    )
    
def on_replay_file_selected(sender, app_data, user_data):
    world = user_data['world']
    # 'app_data' contains the selected file path and name
    replay_file_path = app_data['file_path_name']
    if replay_file_path:
        # Just save the file path without loading the replay
        world.selected_replay_file = replay_file_path
        world.hud.notification(f"Replay file '{replay_file_path}' selected. Press 'Load Replay' to proceed.")
        dpg.set_value(user_data['replay_file_path_tag'], replay_file_path)
    else:
        dpg.set_value(user_data['replay_file_path_tag'], "No directory selected.")
        world.hud.notification("No replay file selected.")

def on_load_replay(sender, app_data, user_data):
    world, client = user_data['world'], user_data['client']
    replay_file_path = world.selected_replay_file
    if replay_file_path:
        world.is_replay_mode = True
        vehicle_id = world.vehicle_actors[world.current_index].id if not world.is_spectator_mode else 0
        client.replay_file(replay_file_path, world.recording_start, 0, vehicle_id)
        world.hud.notification(f"Loading replay file '{replay_file_path}'.")
    else:
        world.hud.notification("No replay file has been selected yet.")

def setup_replay_file_dialog(world, client, replay_file_path_tag):
    # Define the callback user_data as a dictionary with world and client
    user_data = {'world': world, 'client': client, 'replay_file_path_tag': replay_file_path_tag}

    # Check if the replay file dialog already exists and delete it if it does
    if dpg.does_item_exist("replay_file_dialog_id"):
        dpg.delete_item("replay_file_dialog_id")

    # Add the file dialog with the intention to select files, not directories
    with dpg.file_dialog(
        label="Select Replay File",
        callback=on_replay_file_selected,
        show=False,
        height=400,
        tag="replay_file_dialog_id",
        user_data=user_data  # Pass the user_data to the callback
    ):
        # Filter for specific file extensions
        dpg.add_file_extension(".log", color=(255, 255, 255, 255))  # Assuming .log is the extension for replay files


def on_browse_replay_file(sender, app_data, user_data):
    # Show the file dialog for selecting the replay file
    dpg.show_item("replay_file_dialog_id")

def delete_item_if_exists(tag):
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

def run_gui(world, client):
    dpg.create_context()

    # Define the tags for file name and file path outside setup_file_dialog
    file_name_tag = "file_name_tag"  # Use a string tag for the file name input
    file_path_tag = "file_path_tag"  # Use a string tag for the file path display

    delete_item_if_exists(file_name_tag)
    delete_item_if_exists(file_path_tag)

    # Create the user_data dictionary before calling setup_file_dialog
    user_data = {'file_name_tag': file_name_tag, 'file_path_tag': file_path_tag}

    # Call setup_file_dialog with user_data as an argument
    setup_file_dialog(user_data)

    with dpg.window(label="CARLA Simulator Control Panel", no_move=True, no_resize=True, width=600, height=1000):
        dpg.add_text("Control Panel for CARLA Simulator")
        dpg.add_button(label="Restart Simulation", callback=on_button_click, user_data=world)

        with dpg.collapsing_header(label="Vehicle Settings", default_open=False):
            # Create an input field for the number of vehicles
            vehicle_input_tag = dpg.add_input_int(label="Number of Vehicles", default_value=world.default_num_vehicles)
            # Add a button to spawn the vehicles with the input number
            dpg.add_button(label="Spawn Vehicles", callback=on_spawn_vehicles, user_data=(world, vehicle_input_tag))

            # Create a slider for the traffic manager conditions
            traffic_level_tag = dpg.add_slider_int(label="Traffic Conditions Level", default_value=0, min_value=0, max_value=10)
            dpg.add_button(label="Apply Traffic Conditions", callback=on_apply_traffic_conditions, user_data=(world, traffic_level_tag))
            dpg.add_button(label="Toggle Autopilot", callback=on_toggle_autopilot, user_data=world)
            dpg.add_button(label="Toggle World Freeze", callback=on_toggle_freeze, user_data=world)

        with dpg.collapsing_header(label="World Settings", default_open=False):
            pedestrian_input_tag = dpg.add_input_int(label="Number of Pedestrians", default_value=25)
            dpg.add_button(label="Spawn Pedestrians", callback=on_apply_pedestrians, user_data=(world, pedestrian_input_tag))
            props_input_tag = dpg.add_input_int(label="Number of Props", default_value=75)
            dpg.add_button(label="Spawn Props", callback=on_apply_props, user_data=(world, props_input_tag))

        with dpg.collapsing_header(label="Recording Controls", default_open=False):
            # Input for file name
            dpg.add_input_text(tag=file_name_tag, label="File Name", default_value="my_recording.log")
            # Display the full path in a read-only input text
            dpg.add_input_text(tag=file_path_tag, label="Full Path", default_value="", readonly=True)
            # Button to browse for folder
            dpg.add_button(label="Browse Location", callback=on_browse, user_data=user_data)
            dpg.add_button(label="Start Recording", callback=on_start_recording, user_data=(world, file_path_tag))
            dpg.add_button(label="Stop Recording", callback=on_stop_recording, user_data=world)
            replay_file_path_tag = dpg.add_input_text(label="Selected Replay File", default_value="", readonly=True)
            setup_replay_file_dialog(world, client, replay_file_path_tag)
            dpg.add_button(label="Browse Replay File", callback=on_browse_replay_file, user_data={'replay_file_path_tag': replay_file_path_tag, 'world': world, 'client': client})
            dpg.add_button(label="Load Replay", callback=on_load_replay, user_data={'world': world, 'client': client})

        with dpg.collapsing_header(label="More Controls", default_open=False):
            dpg.add_text("More Control Elements")
            dpg.add_button(label="Toggle Spectator Mode", callback=toggle_spectator_mode, user_data=world)
            dpg.add_button(label="Spectator to Closest Vehicle", callback=on_spectator_to_closest_vehicle, user_data=world)

            # Horizontal group for side-by-side buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="Previous Vehicle", callback=on_switch_to_previous_vehicle, user_data=world)
                dpg.add_button(label="Next Vehicle", callback=on_switch_to_next_vehicle, user_data=world)
            


    dpg.create_viewport(title="CARLA Simulator GUI", width=600, height=1000)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Start the Dear PyGui event loop
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()




# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args, client, controller=None):
        self.world = carla_world
        self.client = client
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.bp_lib = self.world.get_blueprint_library() 
        self.spawn_points = self.world.get_map().get_spawn_points() 
        self.is_replay_mode = False
        self.is_spectator_mode = False
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.controller = controller
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.vehicle_actors = []
        self.default_num_vehicles = 100
        self.restart_requested = False
        self.is_restart = False
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.vehicle_id = 0
        self.vehicle_autopilot_states = {}
        self.spawned_props = []
        self.moving_props = []
        self.walker_controllers = []
        self.current_index = 0 # Start with the first vehicle in the list
        self.camera_offset = carla.Location(x=-4, z=3)
        self.continue_loop = True
        self.freeze = False
        self.spectator_location = None
        self.tm_random = random.Random()
        self.tm_random.seed(0)
        self.prop_random = random.Random()
        self.prop_random.seed(0)
        self.step_size = 0.03 # adjust this value as needed
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def set_controller(self, controller):
        self.controller = controller

    def spawn_vehicles(self, number_of_vehicles=None):
        if number_of_vehicles is None:
            number_of_vehicles = self.default_num_vehicles

        spawned_vehicles = 0
        for _ in range(number_of_vehicles):
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(self.spawn_points))
            if npc is not None:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.vehicle_autopilot_states[npc.id] = True
                self.vehicle_actors.append(npc)
                spawned_vehicles += 1

        # Remove the player vehicle if it's in the list
        non_player_vehicles = [actor for actor in self.vehicle_actors if 'vehicle' in actor.type_id and actor != self.player]

        # Update the list with non-player vehicles
        self.vehicle_actors = non_player_vehicles

        self.hud.notification(f"Spawned {spawned_vehicles} Vehicles")
        print("current index: " + str(self.current_index))

    def destroy_vehicles(self):
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()

    def switch_to_next_vehicle(self):
        if self.vehicle_actors:  # check if the list is not empty
            self.current_index = (self.current_index + 1) % len(self.vehicle_actors)  # loop back to the start if at the end of the list
            print("current index: " + str(self.current_index))

    def switch_to_previous_vehicle(self):
        if self.vehicle_actors:  # check if the list is not empty
            self.current_index = (self.current_index - 1) % len(self.vehicle_actors)  # loop back to the end if at the start of the list
            print("current index: " + str(self.current_index))
    
    def toggle_autopilot(self):
        if self.controller:
            self.controller.toggle_autopilot()

    def spawn_pedestrians(self, number_of_pedestrians=25):
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        for _ in range(number_of_pedestrians):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc

                try:
                    # Spawn a walker
                    walker_bp = random.choice(blueprints_walkers)
                    walker = self.world.spawn_actor(walker_bp, spawn_point)
                    
                    # Spawn the WalkerAIController and attach it to the walker
                    walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    walker_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)

                    self.walker_controllers.append(walker_controller)

                    # Start the WalkerAIController
                    walker_controller.start()
                    target_location = self.world.get_random_location_from_navigation()
                    walker_controller.go_to_location(target_location)
                    walker_controller.set_max_speed(2.0)  # Set speed

                except RuntimeError as e:
                    print(f"Failed to spawn pedestrian at {spawn_point.location}. Error: {e}")
        self.hud.notification(f"Spawned {number_of_pedestrians} Pedestrians")

    def spawn_props(self, number_of_props=75):
        prop_blueprints = self.world.get_blueprint_library().filter('static.prop.*')
        waypoints = self.map.generate_waypoints(distance=2.0)

        # Use the provided number_of_props parameter to decide how many props to spawn
        for _ in range(number_of_props):
            random_bp = self.prop_random.choice(prop_blueprints)
            random_waypoint = self.prop_random.choice(waypoints)
            spawn_location = random_waypoint.transform.location
            spawn_location.z += self.prop_random.uniform(5, 10)
            
            spawn_transform = carla.Transform(spawn_location)
            bin_prop = self.world.spawn_actor(random_bp, spawn_transform)
            
            if bin_prop:
                self.spawned_props.append(bin_prop)
                physics_command = carla.command.SetSimulatePhysics(bin_prop.id, True)
                self.client.apply_batch([physics_command])

        self.hud.notification(f"Spawned {number_of_props} Props")

    def tm_conditions(self, level):
        self.tm_random.seed(0)
        if level == 0:
            for v in self.vehicle_actors:
                self.traffic_manager.vehicle_percentage_speed_difference(v, 0)
                self.traffic_manager.ignore_lights_percentage(v,0)
                self.traffic_manager.distance_to_leading_vehicle(v,5)
        else:
                num_to_affect = len(self.vehicle_actors) * level // 10
                vehicles_to_affect = random.sample(self.vehicle_actors, num_to_affect)
                for v in vehicles_to_affect:
                    self.traffic_manager.vehicle_percentage_speed_difference(v, random.randint(-100, 0))
                    self.traffic_manager.ignore_lights_percentage(v, random.randint(0, 100))
                    self.traffic_manager.distance_to_leading_vehicle(v, random.randint(0, 5))
        self.hud.notification(f"Traffic Conditions Level: {level}")

    def request_restart(self):
        self.restart_requested = True

    def restart_if_requested(self):
        if self.restart_requested:
            self.restart()
            self.restart_requested = False

    def restart(self):
        self.is_restart = True
        # Destroy all vehicles and pedestrians
        for actor in self.world.get_actors():
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                if self.player is None or actor.id != self.player.id:  # Ensure it's not the hero vehicle
                    actor.destroy()
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.vehicle_actors.clear()
            self.spawned_props.clear()
            self.destroy()
            self.destroy_props()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.vehicle_actors.append(self.player)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
            self.is_restart = False
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
            self.vehicle_actors.append(self.player)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self.is_restart = False

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def move_props(self, speed=0.1):
        for prop in self.moving_props:
            current_location = prop.get_location()
            target_location = prop.target_location
            # calculate new_location based on speed and direction to the target_location
            if current_location.distance(target_location) > speed:
                direction = target_location - current_location
                direction = direction / direction.length() * speed  # normalize and multiply by speed
                new_location = current_location + direction
                prop.set_location(new_location)
            else:
                prop.set_location(target_location)
                self.moving_props.remove(prop)  # remove prop from moving_props when it reaches the target        

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy_props(self):
        for prop in self.spawned_props:
            prop.destroy()
        self.spawned_props = []  # Clear the list after destroying all props

    def destroy_pedestrians(self):
        for actor in self.world.get_actors().filter('*walker.pedestrian.*'):
            actor.destroy()

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

    def spectator_to_closest_vehicle(self):
        self.spectator_location = self.world.get_spectator().get_location()

        closest_vehicle = None
        min_distance = float('inf')

        for vehicle in self.vehicle_actors:
            distance = vehicle.get_location().distance(self.spectator_location)
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vehicle

        if closest_vehicle:
            self.current_index = self.vehicle_actors.index(closest_vehicle)
            print(closest_vehicle)
            print('current index: ', self.current_index)

        self.is_spectator_mode = False

    def pause_vehicles(self):
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id in self.vehicle_autopilot_states:
                vehicle.set_autopilot(False)
                control = vehicle.get_control()
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                vehicle.apply_control(control)

    def resume_vehicles(self):
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id in self.vehicle_autopilot_states and self.vehicle_autopilot_states[vehicle.id]:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())

    def reset_world(self, hero_vehicle):
    # Iterate through all vehicle actors
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id != hero_vehicle.id:  # Check if it's not the hero vehicle
                vehicle.destroy()

        # Iterate through all pedestrian actors
        for pedestrian in self.world.get_actors().filter('walker.pedestrian.*'):
            pedestrian.destroy()



# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self.world = world
        self._autopilot_enabled = start_in_autopilot
        self.sync_mode = False
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        self.sync_mode = sync_mode
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    world.destroy_vehicles()
                    world.destroy_props()
                    world.destroy_pedestrians()
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.load_map_layer()
                elif event.key == K_b and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.is_replay_mode == False:
                        world.switch_to_previous_vehicle()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.next_sensor()
                elif event.key == K_n and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.is_replay_mode == False:
                        world.switch_to_next_vehicle()
                elif event.key == K_f and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.freeze == False:
                        world.hud.notification("World Frozen")
                        world.pause_vehicles()
                        world.freeze = True
                    else:
                        world.hud.notification("World Unfrozen")
                        world.resume_vehicles()
                        world.freeze = False
                elif event.key == K_k:
                    world.spawn_pedestrians()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_j:
                    world.spawn_vehicles()
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                # elif event.key > K_0 and event.key <= K_9:
                #     index_ctrl = 0
                #     if pygame.key.get_mods() & KMOD_CTRL:
                #         index_ctrl = 9
                #     world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_0:
                    world.tm_conditions(0)
                elif event.key == K_1:
                    world.tm_conditions(1)
                elif event.key == K_2:
                    world.tm_conditions(2)
                elif event.key == K_3:
                    world.tm_conditions(3)
                elif event.key == K_4:
                    world.tm_conditions(4)
                elif event.key == K_5:
                    world.tm_conditions(5)
                elif event.key == K_6:
                    world.tm_conditions(6)
                elif event.key == K_7:
                    world.tm_conditions(7)
                elif event.key == K_8:
                    world.tm_conditions(8)
                elif event.key == K_9:
                    world.tm_conditions(9)
                elif event.key == K_MINUS:
                    world.tm_conditions(10)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    # world.camera_manager.toggle_recording()
                    if not self._ackermann_enabled:
                        world.hud.notification("Reverse Gear On")
                        self._control.gear = 1 if self._control.reverse else -1
                    else:
                        world.hud.notification("Reverse Gear Off")
                        self._ackermann_reverse *= -1
                        # Reset ackermann control
                        self._ackermann_control = carla.VehicleAckermannControl()                   
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("C:/Users/saaso/Desktop/Grad School/Spring 2023/Research Asst/CARLA/recording01.log")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    world.is_replay_mode = True
                    # work around to fix camera at start of replaying
                    # current_index = world.camera_manager.index
                    # world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'recording01.log'")
                    # replayer
                    if world.is_spectator_mode == False:
                        client.replay_file("C:/Users/saaso/Desktop/Grad School/Spring 2023/Research Asst/CARLA/recording01.log", world.recording_start, 0, world.vehicle_actors[world.current_index].id)
                    else:
                        client.replay_file("C:/Users/saaso/Desktop/Grad School/Spring 2023/Research Asst/CARLA/recording01.log", world.recording_start, 0, 0)
                    # world.camera_manager.set_sensor(current_index)
                elif event.key == K_y:
                    if world.is_replay_mode == False:
                        world.is_replay_mode = True
                        world.hud.notification("Replay mode is ON")
                    else:
                        world.is_replay_mode = False
                        world.hud.notification("Replay mode is OFF")
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_q and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.is_spectator_mode == False:
                        world.hud.notification("Spectator Mode is ON")
                        world.is_spectator_mode = True
                    else:
                        world.hud.notification("Spectator Mode is OFF")
                        world.is_spectator_mode = False
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f and not (pygame.key.get_mods() & KMOD_CTRL):
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q and not (pygame.key.get_mods() & KMOD_CTRL):
                        world.spectator_to_closest_vehicle()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l:
                        print(world.player.get_transform())

                    # elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                    #     current_lights ^= carla.VehicleLightState.Special1
                    # elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                    #     current_lights ^= carla.VehicleLightState.HighBeam
                    # elif event.key == K_l:
                    #     # Use 'L' key to switch between lights:
                    #     # closed -> position -> low beam -> fog
                    #     if not self._lights & carla.VehicleLightState.Position:
                    #         world.hud.notification("Position lights")
                    #         current_lights |= carla.VehicleLightState.Position
                    #     else:
                    #         world.hud.notification("Low beam lights")
                    #         current_lights |= carla.VehicleLightState.LowBeam
                    #     if self._lights & carla.VehicleLightState.LowBeam:
                    #         world.hud.notification("Fog lights")
                    #         current_lights |= carla.VehicleLightState.Fog
                    #     if self._lights & carla.VehicleLightState.Fog:
                    #         world.hud.notification("Lights off")
                    #         current_lights ^= carla.VehicleLightState.Position
                    #         current_lights ^= carla.VehicleLightState.LowBeam
                    #         current_lights ^= carla.VehicleLightState.Fog
                    # elif event.key == K_i and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    #     garbage_bin_bp = world.world.get_blueprint_library().find('static.prop.bin')
                    #     if garbage_bin_bp:
                    #         spawn_transform = carla.Transform(carla.Location(x=-119.944664, y=10.201465, z=0.151099), carla.Rotation(pitch=-0.007506, yaw=-11.170398, roll=0.000788))
                    #         bin_prop = world.world.spawn_actor(garbage_bin_bp, spawn_transform)
                    #         if bin_prop:
                    #             world.spawned_props.append(bin_prop)
                    #             bin_prop.target_location = carla.Location(x=-110.168854, y=9.182446, z=0.000392)
                    #             world.moving_props.append(bin_prop)
                    #             print(f"Spawned garbage bin prop at {spawn_transform.location}")
                    #         else:
                    #             print("Failed to spawn garbage bin prop.")
                    #     else:
                    #         print("Could not find the blueprint for the garbage bin prop.")
                    elif event.key == K_i and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        world.spawn_props()
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    def toggle_autopilot(self):
        if not self._autopilot_enabled and not self.sync_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                "experience some issues with the traffic simulation")
        self._autopilot_enabled = not self._autopilot_enabled
        self.world.player.set_autopilot(self._autopilot_enabled)
        self.world.hud.notification(
            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off')) 

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        spectator = sim_world.get_spectator()

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args, client)
        controller = KeyboardControl(world, args.autopilot)
        world.set_controller(controller)


        gui_thread = threading.Thread(target=run_gui, args=(world,client))
        gui_thread.daemon = True
        gui_thread.start()

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            else:
                sim_world.wait_for_tick()  # This ensures the world updates in async mode

            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            
            if not world.is_spectator_mode and world.vehicle_actors and not world.is_restart:
                vehicle = world.vehicle_actors[world.current_index]
                vehicle_transform = vehicle.get_transform()
                rotated_offset = rotate_vector(world.camera_offset, vehicle_transform.rotation)
                target_location = vehicle_transform.location + rotated_offset
                current_location = spectator.get_location()
                new_location = carla.Location(
                    x=lerp(current_location.x, target_location.x, world.step_size),
                    y=lerp(current_location.y, target_location.y, world.step_size),
                    z=lerp(current_location.z, target_location.z, world.step_size)
                )
                new_transform = carla.Transform(new_location, vehicle_transform.rotation)
                spectator.set_transform(new_transform)
            
            world.restart_if_requested()
            world.move_props()
            world.tick(clock)

            world.render(display)
            pygame.display.flip()
    finally:
        if original_settings:
            sim_world.apply_settings(original_settings)
        if world and world.recording_enabled:
            client.stop_recorder()
        if world:
            world.destroy()
        pygame.quit()



# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
