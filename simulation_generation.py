import openai
import torch
from openai import OpenAI
import base64
import requests
from copy import deepcopy
 

 SYSTEM_PROMPT = \
"""I am the planning system of an interactive manipulator robot operating to fulfill human desires. Given 1) an instruction from a human user, 2) a description of the scene, 3) a robot task plan, I will produce code in the form of Python functions to represent the human's preferences. The preference functions I produce will be used by the robot's motion planner to generate trajectories that are preferable for the human user.

Definitions:
- A manipulation primitive is an action that the robot can execute. Its motion is determined by a set of continuous parameters.
- The motion planner generates feasible trajectories in the form of continuous parameters for manipulation primitives in sequence.
- A task plan is a sequence of manipulation primitives that the robot should execute.

The robot can perceive the states of objects in the environment and their relation to each other.

The robot can detect the following states of individual objects:
- Free(a): Object a is free to be picked up.
- Ingripper(a): Object a is in the robot's gripper.

The robot can detect the following relationships among objects:
- On(a, b): Object a is on object b.

The robot has access to the following manipulation primitives:
- Pick(a): The robot picks up object a.
- Place(a, b): The robot places object a on object b.
- Static_handover(a, b): The robot hands over object a to a human hand b.

Objective:
- I will only output preference functions for primitives in the task plan if strictly necessary to fulfill the user's instruction. Otherwise, I will output None.
- I will then produce a preference function for the manipulation primitives in the task plan that require it.
- The preference functions will output the probability that a human partner would be satisfied with the generated motion for each manipulation primitive in the task plan.

Object relation definitions:
- We consider distances of 5cm as close.
- We refer to two objects as close if their centers have a distance of at least 10cm.
- We refer to two objects as far starting at 20cm, but ideally 100cm.
- For rotations, we consider differences of torch.pi/6 as small, but ideally torch.pi/8.

I will format the preference functions as Python functions of the following signature:

```
def {{FunctionName}}Fn(
    state: torch.Tensor,
    action: torch.Tensor, 
    next_state: torch.Tensor, 
    primitive: Optional[Primitive] = None, 
    env: Optional[Environment] = None
) -> torch.Tensor:
    r\"\"\"Evaluates the preference probability of {{PreferenceDescription}}.
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
    # Your code here
```

# Helper functions
I will use helper functions to implement the preference functions.
Most preference functions will follow the same structure:
1. Extract the object poses and relations from the state and primitive.
2. Calculate one or more metrics to evaluate the preference on.
3. Calculate the probability that the action satisfies the human's preferences based on the metrics.
4. Combine the probabilities.
5. Return the total probability.


## Object poses and relations
```
get_object_id_from_name(name: str, env: Env, primitive: Primitive) -> int:
\"\"\"Return the object identifier from a given object name.\"\"\"
```

```
get_object_id_from_primitive(arg_id: int, primitive: Primitive) -> int:
\"\"\"Return the object identifier from a primitive and its argument id.
Example: The primitive `Pcik` has one argument ids: `object` with `arg_id = 0`.
Example: The primitive `Place` has two argument ids: `object` with `arg_id = 0` and `target` with `arg_id = 1`.
Example: The primitive `Static_handover` has two argument ids: `object` with `arg_id = 0` and `target_body_part` with `arg_id = 1`.
\"\"\"
```

```
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
```

```
def generate_pose_batch(pose: Sequence[float], pose_like: torch.Tensor) -> torch.Tensor:
\"\"\"Repeat a pose for the batch dimension.

Example: generate the origin pose [0, 0, 0, 1, 0, 0, 0] to be used as an argument of type `torch.Tensor` in later functions.

Args:
    pose: the pose to repeat in [x, y, z, qw, qx, qy, qz] format.
    pose_like: Another pose tensor to get the batch size and device from.
Returns:
    The pose in shape [batch_size, 7].
\"\"\"
```

```
def build_direction_vector(pose_1: torch.Tensor, pose_2: torch.Tensor) -> torch.Tensor:
\"\"\"Build the vector pointing from pose 1 to pose 2.\"\"\"
```

## Metric functions
```
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
```

```
def position_diff_along_direction(
    pose_1: torch.Tensor, pose_2: torch.Tensor, direction: Union[torch.Tensor, Sequence[float]] = [1, 0, 0]
) -> torch.Tensor:
\"\"\"Calculate the positional difference of two poses along the given direction.

Example: can be used to evaluate if an object is placed left or right of another object.
Returns the dot product of the positional difference and the given direction.

Args:
    pose_{1, 2}: the poses of the two objects.
    direction: the direction along which to calculate the difference.
        Can be either a tensor of shape [..., 3] with one direction vector per entry in the batch or a list indicating
        a single direction vector that is used for all entries in the batch.
Returns:
    The positional difference in shape [..., 1]
\"\"\"
```

```
def position_metric_normal_to_direction(
    pose_1: torch.Tensor, pose_2: torch.Tensor, direction: Union[torch.Tensor, Sequence[float]] = [1, 0, 0]
) -> torch.Tensor:
\"\"\"Calculate the positional difference of two poses normal to the given direction.

Given a point (pose_1), the function calculates the distance to a line defined by a point (pose_2) and a direction.

Args:
    pose_{1, 2}: the poses of the two objects.
    direction: the direction normal to which to calculate the difference.
        Can be either a tensor of shape [..., 3] with one direction vector per entry in the batch or a list indicating
        a single direction vector that is used for all entries in the batch.
Returns:
    The positional difference in shape [..., 1]
\"\"\"
```

```
great_circle_distance_metric(pose_1: torch.Tensor, pose_2: torch.Tensor) -> torch.Tensor:
\"\"\"Calculate the difference in orientation in radians of two poses using the great circle distance.

Assumes that the position entries of the poses are direction vectors `v1` and `v2`.
The great circle distance is then `d = arccos(dot(v1, v2))` in radians.
\"\"\"
```

```
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
```

```
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
```

## Probability functions
```
threshold_probability(metric: torch.Tensor, threshold: float, is_smaller_then: bool = True) -> torch.Tensor:
\"\"\"If `is_smaller_then`: return `1.0` if `metric < threshold` and `0.0` otherwise.
If not `is_smaller_then`: return `1.0` if `metric >= threshold` and `0.0` otherwise.
\"\"\"
```

```
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
```

```
def normal_probability(metric: torch.Tensor, mean: float, std_dev: float, is_smaller_then: bool = True) -> torch.Tensor:
\"\"\"Return a probability function based on the cummulative distribution function with `mean` and `std_dev`.

This should only be used if the metric has no point from which onward the probability should be `0.0`.
Most of the time, we would use the threshold or linear probability functions.

Args:
    metric: the metric to calculate the value for.
    mean: the mean of the cummulative distribution function.
    std_dev: the standard deviation of the cummulative distribution function.
    is_smaller_then: if true, invert the return value with `(1-p)`.
Returns:
    `cdf(metric, mean, std_dev)`
\"\"\"
```

```
probability_intersection(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
\"\"\"Calculate the intersection of two probabilities `p = p_1 * p_2`.\"\"\"
```

```
probability_union(p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
\"\"\"Calculate the union of two probabilities `p = max(p_1, p_2)`.\"\"\"
```
"""



TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(cyan_box, table)
2. Place(cyan_box, table)
"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
Make sure that the red box, the blue box, and the cyan box are in a line on the table.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - PlaceInLineWithRedAndBlueBoxFn

```
def PlaceInLineWithRedAndBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed in line with the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, red_to_blue)
    # The x difference should be as small as possible but no larger than 5cm (close).
    lower_threshold = 0.0
    upper_threshold = 0.05
    probability = linear_probability(normal_distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability
```
"""


TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(red_box, table)
2. Place(red_box, table)
3. Pick(cyan_box, table)
4. Place(cyan_box, table)
"""

EXAMPLE_USER1 = \
f"""{TASK_PREFIX}

Instruction:
Make sure that the red box is placed close to the blue box and that the cyan box is placed far away from both the red and the blue box.
"""

EXAMPLE_ASSISTANT1 = \
"""
custom_fns:
  - null
  - PlaceNextToBlueBoxFn
  - null
  - PlaceFarAwayFromRedAndBlueFn

```
def PlaceNextToBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed next to the blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from

    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant.
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed at least 10cm (close) next to the blue box.
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    close_by_probability = linear_probability(distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return close_by_probability
```

```
def PlaceFarAwayFromRedAndBlueFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed far away from the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed far away from the red box
    distance_metric_red = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.20
    upper_threshold = 1.0
    far_away_probability_red = linear_probability(
        distance_metric_red, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Evaluate if the object is placed far away from the blue box
    distance_metric_blue = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    far_away_probability_blue = linear_probability(
        distance_metric_blue, lower_threshold, upper_threshold, is_smaller_then=False
    )
    # Combine the two probabilities
    total_probability = probability_intersection(far_away_probability_red, far_away_probability_blue)
    return total_probability
```
"""

TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(blue_box, table)
2. Place(blue_box, table)
3. Pick(cyan_box, table)
4. Place(cyan_box, table)
"""

EXAMPLE_USER2 = \
f"""{TASK_PREFIX}

Instruction:
Arrange the red box, blue box, and cyan box in a triangle of edge length 20 cm.
"""

EXAMPLE_ASSISTANT2 = \
"""
custom_fns:
  - null
  - PlaceNextToRedBox20cmFn
  - null
  - PlaceNextToRedBoxAndBlueBox20cmFn
```
def PlaceNextToRedBox20cmFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluate if the object is placed 20cm next to the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed 20cm next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15
    ideal_point = 0.20
    upper_threshold = 0.25
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
```

```
def PlaceNextToRedBoxAndBlueBox20cmFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluate if the object is placed 20cm next to the red and blue box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed 20cm next to the red box.
    distance_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.15
    ideal_point = 0.20
    upper_threshold = 0.25
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    probability_red_box = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    distance_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    probability_blue_box = probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
    return probability_intersection(probability_red_box, probability_blue_box)
```
"""


# 1. baseline: text only
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
    },


    {
        "role": "system",
        "name": "example_user",
        "content": EXAMPLE_USER1
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT1
    },


    {
        "role": "system",
        "name": "example_user",
        "content": EXAMPLE_USER2
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT2
    }
]


TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(blue_box, table)
2. Place(blue_box, table)
3. Pick(cyan_box, table)
4. Place(cyan_box, table)
"""

instruction = "Make sure that the blue box and the cyan box are left of the red box."

EXAMPLE_USER_EVAL = \
f"""{TASK_PREFIX}

Instruction:
{instruction}
"""

# Instruction and task-related user input

prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "name": "user_instruction",
        "content": EXAMPLE_USER_EVAL +\
            "Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable)."
    }
)
prompt.append(
    {
        "role": "user",
        "name": "rules",
        "content": "1. Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable).\n\
                    2. If an action does not need a custom function, add a `- null` entry to the custom_fns list to make sure the list is complete.\n\
                    3. Use the `linear_probability()` function over the `threshold_probability()` or `normal_probability()` functions when possible to improve performance of the planner."
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


# 2. VL General:
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image

image_path0 = "/Users/ch/Documents/my_folder/resources/school/enrollment/enroll/course_schedule/fall_2024_courses/16824/project/image2interaction/images/left_boxes.png"

image_path1 = "/Users/ch/Documents/my_folder/resources/school/enrollment/enroll/course_schedule/fall_2024_courses/16824/project/image2interaction/images/line_box.jpeg"

image_path2 = "/Users/ch/Documents/my_folder/resources/school/enrollment/enroll/course_schedule/fall_2024_courses/16824/project/image2interaction/images/red_box_bottom.jpeg"
image_path3 = "/Users/ch/Documents/my_folder/resources/school/enrollment/enroll/course_schedule/fall_2024_courses/16824/project/image2interaction/images/red_on.jpeg"
image_path4 = "/Users/ch/Documents/my_folder/resources/school/enrollment/enroll/course_schedule/fall_2024_courses/16824/project/image2interaction/images/left_boxes.png"


# Getting the base64 string
# base64_image = encode_image(image_path)

base64_image0 = encode_image(image_path0)

base64_image1 = encode_image(image_path1)
base64_image2 = encode_image(image_path2)
base64_image3 = encode_image(image_path3)
base64_image4 = encode_image(image_path3)

TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(blue_box, table)
2. Place(blue_box, table)
3. Pick(cyan_box, table)
4. Place(cyan_box, table)
"""

EXAMPLE_USER_image = \
f"""{TASK_PREFIX}

Instruction:
A visual demonstratin of the correct handover behavior is in the image.
"""

EXAMPLE_ASSISTANT_image = \
"""
custom_fns:
  - null
  - LeftOfRedBoxFn
  - null
  - LeftOfRedBoxFn

```
def LeftOfRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluate if the object is placed left of the red box.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    # Evaluate if the object is placed left of the red box
    left = [0.0, -1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, red_box_pose, left)
    lower_threshold = 0.0
    # The direction difference should be positive if the object is placed left of the red box.
    is_left_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    return is_left_probability
```
"""


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
            "url": f"data:image/jpeg;base64,{base64_image0}"
          }
        }]
    },

    {
        "role": "system",
        "name": "example_assistant",
        "content": EXAMPLE_ASSISTANT_image
    }
]




# Instruction and task-related user input
TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(cyan_box, table)
2. Place(cyan_box, table)
"""

instruction = "Visual demonstratin of correct behavior is shown in the first image as below, and demonstration of three wrong behaviors are shown in the second, third and forth images as below.  Please first figure out and describe: 1) what should be the correct behavior? and 2) what should be wrong behaviors? Then complete your task.  The function you generated should directly evaluate both factors as you figured out: reward the correct behavior, and penalize the wrong behavior, meaning if a pose is similar to the wrong behavior, its penalization should be reflected in your preference function generated."

EXAMPLE_USER_EVAL_image = \
f"""{TASK_PREFIX}

Instruction:
{instruction}
"""

# Instruction and task-related user input

prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "name": "user_instruction",
        "content": [
                  {
          "type": "text",
          "text": EXAMPLE_USER_EVAL_image +\
            "Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable)."
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
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image3}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image4}"
          }
        },
        ]
    }
)
prompt.append(
    {
        "role": "user",
        "name": "rules",
        "content": "1. Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable).\n\
                    2. If an action does not need a custom function, add a `- null` entry to the custom_fns list to make sure the list is complete.\n\
                    3. Use the `linear_probability()` function over the `threshold_probability()` or `normal_probability()` functions when possible to improve performance of the planner."
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


#  4. graph of thoughts


# Instruction and task-related user input
TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(cyan_box, table)
2. Place(cyan_box, table)
"""

instruction = "Visual demonstratin of correct behavior is shown in the first image as below, and demonstration of three wrong behaviors are shown in the second, third and forth images as below.  Please first figure out and describe: 1) what should be the correct behavior? and 2) what should be wrong behaviors? Second, please provide your graph of thought with explicitly describing nodes and edge for the correct behavior. Then complete your task.  The function you generated should directly evaluate both factors as you figured out: reward the correct behavior, and penalize the wrong behavior, meaning if a pose is similar to the wrong behavior, its penalization should be reflected in your preference function generated."

EXAMPLE_USER_EVAL_image = \
f"""{TASK_PREFIX}

Instruction:
{instruction}
"""

# Instruction and task-related user input

prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "name": "user_instruction",
        "content": [
                  {
          "type": "text",
          "text": EXAMPLE_USER_EVAL_image +\
            "Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable)."
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
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image3}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image4}"
          }
        },
        ]
    }
)
prompt.append(
    {
        "role": "user",
        "name": "rules",
        "content": "1. Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable).\n\
                    2. If an action does not need a custom function, add a `- null` entry to the custom_fns list to make sure the list is complete.\n\
                    3. Use the `linear_probability()` function over the `threshold_probability()` or `normal_probability()` functions when possible to improve performance of the planner."
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



# Instruction and task-related user input
TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(cyan_box, table)
2. Place(cyan_box, table)
"""

instruction = "Visual demonstratin of correct behavior is shown in the first image as below, and demonstration of three wrong behaviors are shown in the second, third and forth images as below.  Please first figure out and describe: 1) what should be the correct behavior? and 2) what should be wrong behaviors? Second, please provide your graph of thoughts with explicitly describing nodes and edge for: 1) graph of thought for describing what is defined as correct behavior; 2) graph of thoughts for the task for the correct behavior. Then complete your task.  The function you generated should directly evaluate both factors as you figured out: reward the correct behavior, and penalize the wrong behavior, meaning if a pose is similar to the wrong behavior, its penalization should be reflected in your preference function generated."

EXAMPLE_USER_EVAL_image = \
f"""{TASK_PREFIX}

Instruction:
{instruction}
"""

# Instruction and task-related user input

prompt = deepcopy(OPENAI_PROMPT)
prompt.append(
    {
        "role": "user",
        "name": "user_instruction",
        "content": [
                  {
          "type": "text",
          "text": EXAMPLE_USER_EVAL_image +\
            "Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable)."
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
            "url": f"data:image/jpeg;base64,{base64_image2}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image3}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image4}"
          }
        },
        ]
    }
)
prompt.append(
    {
        "role": "user",
        "name": "rules",
        "content": "1. Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable).\n\
                    2. If an action does not need a custom function, add a `- null` entry to the custom_fns list to make sure the list is complete.\n\
                    3. Use the `linear_probability()` function over the `threshold_probability()` or `normal_probability()` functions when possible to improve performance of the planner."
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

