import openai
import torch
from openai import OpenAI
import base64
import requests
from copy import deepcopy
 

SYSTEM_PROMPT = \
"""I am the planning system of an interactive manipulator robot operating alongside humans. Given 1) an instruction from a human user, 2) a description of the scene, 3) a robot task plan, I will produce code in the form of Python functions to represent the human's preferences. The preference functions I produce will be used by the robot's motion planner to generate trajectories that are both feasible in the environment and preferable for the human user.

Definitions:
- A manipulation primitive is an action that the robot can execute. Its motion is determined by a set of continuous parameters
- The motion planner generates feasible trajectories in the form of continuous parameters for manipulation primitives in sequence
- A task plan is a sequence of manipulation primitives that the robot should execute

The robot can perceive the following information about the environment:
- The objects in the environment (including the human)
- The states of individual objects
- The relationships among objects

The robot can detect the following states of individual objects:
- Free(a): Object a is free to be picked up

The robot can detect the following relationships among objects:
- On(a, b): Object a is on object b

The robot has access to the following manipulation primitives:
- Pick(a): The robot picks up object a. Action ranges: [x: [-0.200, 0.200], y: [-0.100, 0.100], z: [-0.070, 0.070], theta: [-0.157, 0.157]]
- Handover(a, b): The robot hands over object a to a human hand b. Action ranges: [pitch: [-2.000, 0.000], yaw: [-3.142, 3.142], distance: [0.400, 0.900], height: [0.200, 0.700]]

Objective:
- I will produce a preference function for each manipulation primitive in the task plan
- The preference functions will output the probability that a human collaborative partner would be satisfied with the generated motion for each manipulation primitive in the task plan


I will format the preference functions as Python functions of the following signature:

def {{Primitive.name}}PreferenceFn(
    state: torch.Tensor,
    action: torch.Tensor, 
    next_state: torch.Tensor, 
    primitive: Optional[Primitive] = None, 
    env: Optional[Environment] = None
) -> torch.Tensor:
    r\"\"\"Evaluates the preference probability of the {{Primitive.name}} primitive.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Predicted next state after executing this action.
        primitive: Optional primitive to receive the object information from
        env: Optional environment to receive the object information from
     Returns:
        The probability that action `a` on primitive {{Primitive.name}} satisfies the preferences of the human partner. 
            Output shape: [batch_size] \in [0, 1].
    \"\"\"
    ...

    
I will use the following helper functions to implement the preference functions:

get_object_id_from_name(name: str, env: Env, primitive: Primitive) -> int:
\"\"\"Return the object identifier from a given object name.\"\"\"

get_object_id_from_primitive(arg_id: int, primitive: Primitive) -> int:
\"\"\"Return the object identifier from a primitive and its argument id.
Example: The primitive `Place` has two argument ids: `object` with `arg_id = 0` and `target` with `arg_id = 1`.
\"\"\"

get_pose(state: torch.Tensor, object_id: int, frame: int = -1) -> torch.Tensor:
\"\"\"Return the pose of an object in the requested frame.

Args:
    state: state (observation) to extract the pose from.
    object_id: number identifying the obect. Can be retrieved with `get_object_id_from_name()` and
        `get_object_id_from_primitive()`.
    frame: the frame to represent the pose in. Default is `-1`, which is world frame. In our simulation, the base
        frame equals the world frame. Give the object id for other frames, e.g., `0` for end effector frame.
Returns:
    The pose in shape [..., 7] with format [x, y, z, qw, qx, qy, qz], with the rotation represented as a quaternion.
\"\"\"

position_norm_metric(
    pose_1: torch.Tensor, pose_2: torch.Tensor, norm: str = "L2", axes: Sequence[str] = ["x", "y", "z"]
) -> torch.Tensor:
\"\"\"Calculate the norm of the positional difference of two poses along the given axes.

Args:
    pose_{1, 2}: the poses of the two objects.
    norm: which norm to calculate. Choose from 'L1', 'L2', and 'Linf'. Defaults to `L2`.
    axes: calculate the norm along the given axes and ignore all other axes. Choose entries from `{'x', 'y', 'z'}`.
Returns:
    The norm in shape [..., 1]
\"\"\"

great_circle_distance_metric(pose_1: torch.Tensor, pose_2: torch.Tensor) -> torch.Tensor:
\"\"\"Calculate the difference in orientation in radians of two poses using the great circle distance.

Assumes that the position entries of the poses are direction vectors `v1` and `v2`.
The great circle distance is then `d = arccos(dot(v1, v2))` in radians.
\"\"\"

pointing_in_direction_metric(
    pose_1: torch.Tensor, pose_2: torch.Tensor, main_axis: Sequence[float] = [1, 0, 0]
) -> torch.Tensor:
\"\"\"Evaluate if an object is pointing in a given direction.

Rotates the given main axis by the rotation of pose_1 and calculates the `great_circle_distance()`
between the rotated axis and pose_2.position.
Args:
    pose_1: the orientation of this pose is used to rotate the `main_axis`.
    pose_2: compare the rotated `main_axis` with the position vector of this pose.
    main_axis: axis describing in which direction an object is pointing in its default configuration.
Returns:
    The great circle distance in radians between the rotated `main_axis` and the position part of `pose_2`.
\"\"\"

rotation_angle_metric(pose_1: torch.Tensor, pose_2: torch.Tensor, axis: Sequence[float]) -> torch.Tensor:
\"\"\"Calculate the rotational difference between pose_1 and pose_2 around the given axis.

Example: The orientation 1 is not rotated and the orientation 2 is rotated around the z-axis by 90 degree.
    Then if the given axis is [0, 0, 1], the function returns pi/2.
    If the given axis is [1, 0, 0], the function returns 0, as there is no rotation around the x-axis.

Args:
    pose_{1, 2}: the orientations of the two poses are used to calculate the rotation angle.
    axis: The axis of interest to rotate around.

Returns:
    The angle difference in radians along the given axis.
\"\"\"

threshold_probability(metric: torch.Tensor, threshold: float, is_smaller_then: bool = True) -> torch.Tensor:
\"\"\"If `is_smaller_then`: return `1.0` if `metric < threshold` and `0.0` otherwise.
If not `is_smaller_then`: return `1.0` if `metric >= threshold` and `0.0` otherwise.
\"\"\"

def linear_probability(
    metric: torch.Tensor, lower_threshold: float, upper_threshold: float, is_smaller_then: bool = True
) -> torch.Tensor:
\"\"\"Return the linear probility given a metric and two thresholds.

If `is_smaller_then` return:
    - `1.0` if `metric < lower_threshold`
    - `0.0` if `metric < upper_threshold`
    - linearly interpolate between 0 and 1 otherwise.
If not `is_smaller_then` return:
    - `1.0` if `metric >= upper_threshold`
    - `0.0` if `metric < lower_threshold`
    - linearly interpolate between 1 and 0 otherwise.
\"\"\"

probability_intersection(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
\"\"\"Calculate the intersection of two probabilities `p = p_1 * p_2`.\"\"\"

probability_union(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
\"\"\"Calculate the union of two probabilities `p = max(p_1, p_2)`.\"\"\""""

TASK_PREFIX = \
"""Detected Objects:
- table: A table with bounding box [3.000, 2.730, 0.050]
- screwdriver: A screwdriver with a rod and a handle
    The screwdriver can be picked both at the rod and the handle
    The handle points in negative x direction `main_axis = [-1, 0, 0]`, the rod in the positive x direction `main_axis = [1, 0, 0]` of the object in the object frame. 
    The object frame has its origin at the point where the rod and the handle meet. 
    The object properties are: handle_length=0.090, rod_length=0.075, handle_radius=0.012, rod_radius=0.003
- left_hand: The human's left hand
- right_hand: The human's right hand 

Object Relationships:
- Free(screwdriver)
- On(screwdriver, table)

Task Plan: 
1. Pick(screwdriver)
2. Handover(screwdriver, right_hand)"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}
Instruction:
Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me and the screwdriver is parallel to the table (z=0)."""


EXAMPLE_ASSISTANT = \
"""def ScrewdriverPickFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates the position of the pick primitive.
c    We want to grasp the screwdirver at the rod, so that the human can easily grab the handle.
e    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
.    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    end_effector_id = get_object_id_from_name("end_effector", env, primitive)
    # Get the pose of the end effector in the object frame
    next_end_effector_pose = get_pose(next_state, end_effector_id, object_id)
    # Assumes the rod length is 0.075 and the rod in the positive x direction in object frame.
    preferred_grasp_pose = torch.FloatTensor([0.075 / 2.0, 0, 0, 1, 0, 0, 0]).to(next_state.device)
    # Calculate the positional norm metric
    position_metric = position_norm_metric(next_end_effector_pose, preferred_grasp_pose, norm="L2", axes=["x"])
    # Calculate the probability
    probability_grasp_handle = threshold_probability(position_metric, 0.075 / 2.0, is_smaller_then=True)
    return probability_grasp_handle

def ScrewdriverHandoverFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates the orientation of the screwdriver handover.
     Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
,    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    current_hand_pose = get_pose(state, hand_id)
    handle_main_axis = [-1.0, 0.0, 0.0]
    # We want to know if the handle is pointing towards the hand position after the handover action.
    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 4.0
    # Calculate the probability
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    return probability_handover_orientation"""

# 1. text only: baseline
headers = {
"Content-Type": "application/json",
"Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
{
    "role": "system",
    "content": SYSTEM_PROMPT
},
{
    "role": "system",
    "name": "example_user",
    "content": EXAMPLE_USER
},
{
    "role": "system",
    "name": "example_assistant",
    "content": EXAMPLE_ASSISTANT
}
]

# Instruction and task-related user input
instruction = "Please make sure the handle points upward or downward when performing the handover. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
{
    "role": "user",
    "content": f"{TASK_PREFIX} / Instruction: {instruction}"
}
)

# API payload
payload = {
"model": "gpt-4o-mini",
"messages": prompt,
"max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
msg_response = response.json()
print(msg_response['choices'][0]['message']['content'])
else:
print(f"Error: {response.status_code}, {response.text}")



# 2. VL-only
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
# image_path = "image2interaction/images/3box_in_line.jpeg"

image_path1 = "image2interaction/images/handover_toward_parallel.jpeg"
image_path2 = "image2interaction/images/handover_toward_vertical_table.jpeg"
image_path3 = "image2interaction/images/wrong.jpeg"


# Getting the base64 string
# base64_image = encode_image(image_path)

base64_image1 = encode_image(image_path1)
base64_image2 = encode_image(image_path2)
base64_image3 = encode_image(image_path3)

EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me and the screwdriver is parallel to the table (z=0). A visual demonstratin of the correct handover behavior is in the image, where where we allow 30 degree angle of the parallel, which is within our error tolerance."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "Please make sure the handle points upward or downward when performing the handover. A visual demonstratin of the correct handover behavior is in the second image. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        # "content": f"{TASK_PREFIX} / Instruction: {instruction}"
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        }]
    }
)

# API payload
payload = {
    "model": "gpt-4o-mini",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")


# 3. VL2I
EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image, where where we allow 30 degree angle of the screwdriver relative to the required direction, which is within our error tolerance."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt
OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "A visual demonstratin of a new correct handover behavior is in this new image. Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, and then complete your task. The function you generated should directly evaluate both factors as you figured out. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        # "content": f"{TASK_PREFIX} / Instruction: {instruction}"
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        }]
    }
)

# API payload
payload = {
    "model": "gpt-4o-mini",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")


EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image, where where we allow 30 degree angle of the screwdriver relative to the required direction, which is within our error tolerance."""

# EXAMPLE_USER_image = \
# f"""{TASK_PREFIX}
# Instruction:
# Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me (the relationship between the object to the human), and the screwdriver is parallel to the table (its direction relative to the table).  A visual demonstratin of the correct handover behavior is in the image."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "A visual demonstratin of a new correct handover behavior is in this new image. Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, and then complete your task. The function you generated should directly evaluate both factors as you figured out. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        # "content": f"{TASK_PREFIX} / Instruction: {instruction}"
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        }]
    }
)

# API payload
payload = {
    "model": "gpt-4o",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image, where where we allow 30 degree angle of the screwdriver relative to the required direction, which is within our error tolerance."""

# EXAMPLE_USER_image = \
# f"""{TASK_PREFIX}
# Instruction:
# Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me (the relationship between the object to the human), and the screwdriver is parallel to the table (its direction relative to the table).  A visual demonstratin of the correct handover behavior is in the image."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "A visual demonstratin of a new correct handover behavior is in this new image. Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, and then complete your task. The function you generated should directly evaluate both factors as you figured out. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        # "content": f"{TASK_PREFIX} / Instruction: {instruction}"
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        }]
    }
)

# API payload
payload = {
    "model": "gpt-4o",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")

EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image."""

# EXAMPLE_USER_image = \
# f"""{TASK_PREFIX}
# Instruction:
# Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me (the relationship between the object to the human), and the screwdriver is parallel to the table (its direction relative to the table).  A visual demonstratin of the correct handover behavior is in the image."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "A visual demonstratin of a new correct handover behavior is in this new image. Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, and then complete your task. The function you generated should directly evaluate both factors as you figured out. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        # "content": f"{TASK_PREFIX} / Instruction: {instruction}"
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        }]
    }
)

# API payload
payload = {
    "model": "gpt-4o",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")

# 4. VL2I-P:
EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image."""

# EXAMPLE_USER_image = \
# f"""{TASK_PREFIX}
# Instruction:
# Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me (the relationship between the object to the human), and the screwdriver is parallel to the table (its direction relative to the table).  A visual demonstratin of the correct handover behavior is in the image."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "A visual demonstratin of a new correct handover pose is in shown in the first image as below, and a wrong pose is shown in the second image as below.  Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, and then complete your task. The function you generated should directly evaluate both factors as you figured out. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        },
        
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image3}"
          }
        },]
    }
)

# API payload
payload = {
    "model": "gpt-4o",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")

EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image."""

# EXAMPLE_USER_image = \
# f"""{TASK_PREFIX}
# Instruction:
# Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me (the relationship between the object to the human), and the screwdriver is parallel to the table (its direction relative to the table).  A visual demonstratin of the correct handover behavior is in the image."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "A visual demonstratin of a new correct handover pose is in shown in the first image as below, and a wrong pose is shown in the second image as below.  Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, 3) what wrong pose it is. Then complete your task.  The function you generated should directly evaluate all three factors as you figured out: reward first two factors, and penalize the 3rd factor, meaning if a pose is similar to the pose in 3), its penalization should be reflected in your preference function generated. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        },
        
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image3}"
          }
        },]
    }
)

# API payload
payload = {
    "model": "gpt-4o",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")


# 5. GVL2I-P:
EXAMPLE_USER_image = \
f"""{TASK_PREFIX}
Instruction:
A visual demonstratin of the correct handover behavior is in the image."""

# EXAMPLE_USER_image = \
# f"""{TASK_PREFIX}
# Instruction:
# Please hand over the screwdriver such that I can easily grab the handle. Make sure that the handle is facing me (the relationship between the object to the human), and the screwdriver is parallel to the table (its direction relative to the table).  A visual demonstratin of the correct handover behavior is in the image."""


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {some_key}"
}


# Define the prompt

OPENAI_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "name": "example_user",
        "content": [
        {
          "type": "text",
          "text": EXAMPLE_USER_image
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT
    }
]

# Instruction and task-related user input
instruction = "Visual demonstratin of correct handover poses are shown in the first image and the second image as below, and a wrong pose is shown in the third image as below.  Please first figure out and describe: 1) the relationship between the screwdriver and the human, and 2) its direction relative to the table, 3) what wrong pose it is. Then complete your task.  The function you generated should directly evaluate all three factors as you figured out: reward first two factors, and penalize the 3rd factor, meaning if a pose is similar to the pose in 3), its penalization should be reflected in your preference function generated. Ignore the Pick preference function for now."
prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "content": [
        {
          "type": "text",
          "text": f"{TASK_PREFIX} / Instruction: {instruction}"
        },

        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image3}"
          }
        },]
    }
)

# API payload
payload = {
    "model": "gpt-4o",
    "messages": prompt,
    "max_tokens": 2048
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    msg_response = response.json()
    print(msg_response['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")