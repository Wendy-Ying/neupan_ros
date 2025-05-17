#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist

class HelmetStatePublisher:
    def __init__(self):
        rospy.init_node('helmet_state_publisher')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.target_frame = "jieting_helmet"
        self.reference_frame = "livox_frame"

        self.last_time = None
        self.last_pos = None

        self.pub = rospy.Publisher("/helmet_state", Twist, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(0.02), self.update)

    def update(self, event):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.target_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )

            t = trans.header.stamp.to_sec()
            pos = trans.transform.translation
            current_pos = np.array([pos.x, pos.y])

            if self.last_time is not None:
                dt = t - self.last_time
                if dt > 0:
                    delta = current_pos - self.last_pos
                    velocity = delta / dt

                    msg = Twist()
                    msg.linear.x = current_pos[0]  # x
                    msg.linear.y = current_pos[1]  # y
                    msg.angular.x = velocity[0]    # vx
                    msg.angular.y = velocity[1]    # vy

                    self.pub.publish(msg)

                    rospy.loginfo("x=%.2f y=%.2f | vx=%.2f vy=%.2f",
                                           current_pos[0], current_pos[1],
                                           velocity[0], velocity[1])

            self.last_time = t
            self.last_pos = current_pos

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF Error: %s", str(e))


if __name__ == "__main__":
    try:
        HelmetStatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
