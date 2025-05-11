#!/usr/bin/env python
import rospy
import tf
import math
from tf.transformations import quaternion_from_euler

def main():
    rospy.init_node('livox_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)

    camera_translation = (-2.666, 2.666, 0.0)
    camera_rotation = quaternion_from_euler(0, 0, -math.pi / 2)

    livox_translation = (0.0, 0.0, 0.0)
    livox_rotation = quaternion_from_euler(0, 0, 0)

    while not rospy.is_shutdown():
        br.sendTransform(
            camera_translation,
            camera_rotation,
            rospy.Time.now(),
            "camera_init",
            "world"
        )

        br.sendTransform(
            livox_translation,
            livox_rotation,
            rospy.Time.now(),
            "livox_frame",
            "os_sensor"
        )

        rate.sleep()

if __name__ == '__main__':
    main()
