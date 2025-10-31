from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dmpc_controller',
            executable='dmpc_node',
            name='dmpc_node',
            output='screen',
            parameters=[{'robot_name': 'puzzlebot1'}]
        )
    ])
