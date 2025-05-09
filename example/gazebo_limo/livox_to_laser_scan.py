#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from livox_ros_driver2.msg import CustomMsg, CustomPoint
import math

class LivoxToLaserScan:
    def __init__(self):
        rospy.init_node('livox_to_laserscan')
        self.angle_min = -math.pi
        self.angle_max = math.pi
        self.angle_increment = math.pi / 250
        self.range_min = 0.1
        self.range_max = 40
        self.z_min = -0.5
        self.z_max = 1.0
        self.num_ranges = int((self.angle_max - self.angle_min) / self.angle_increment)
        self.scan_pub = rospy.Publisher('/limo/scan', LaserScan, queue_size=1)
        rospy.Subscriber('/livox/lidar', CustomMsg, self.callback)

    def callback(self, msg):
        ranges = [float('inf')] * self.num_ranges
        for point in msg.points:
            x = point.x
            y = point.y
            z = point.z

            if not (self.z_min <= z <= self.z_max):
                continue

            r = math.hypot(x, y)
            if self.range_min <= r <= self.range_max:
                angle = math.atan2(y, x)
                if self.angle_min <= angle <= self.angle_max:
                    index = int((angle - self.angle_min) / self.angle_increment)
                    if 0 <= index < self.num_ranges:
                        if r < ranges[index]:
                            ranges[index] = r

        for i in range(len(ranges)):
            if ranges[i] == float('inf'):
                ranges[i] = 0.0

        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = "livox_frame"
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges
        self.scan_pub.publish(scan)

if __name__ == '__main__':
    try:
        LivoxToLaserScan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass