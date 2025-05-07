#!/usr/bin/env python
import rospy
import tf
from tf.transformations import quaternion_from_euler

def main():
    rospy.init_node('livox_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)

    translation = (0.2, 0.0, 0.1)
    rotation = quaternion_from_euler(0, 0, 0)

    while not rospy.is_shutdown():
        br.sendTransform(
            translation,
            rotation,
            rospy.Time.now(),
            "livox_frame",
            "base_link"
        )
        rate.sleep()

if __name__ == '__main__':
    main()
