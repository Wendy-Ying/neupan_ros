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
        self.z_min = -0.4
        self.z_max = 0.4
        self.num_ranges = int((self.angle_max - self.angle_min) / self.angle_increment)
        self.scan_pub = rospy.Publisher('/limo/scan', LaserScan, queue_size=1)
        rospy.Subscriber('/livox/lidar', CustomMsg, self.callback)

    def gaussian_filter(self, data, window_size=5, sigma=0.1):
        filtered = []
        half = window_size // 2

        kernel = [math.exp(-0.5 * ((i - half) / sigma) ** 2) for i in range(window_size)]
        kernel_sum = sum(kernel)
        kernel = [x / kernel_sum for x in kernel]

        for i in range(len(data)):
            smoothed = 0.0
            weight_sum = 0.0
            for j in range(window_size):
                idx = i + j - half
                if 0 <= idx < len(data) and data[idx] != float('inf'):
                    weight = kernel[j]
                    smoothed += data[idx] * weight
                    weight_sum += weight
            if weight_sum > 0:
                filtered.append(smoothed / weight_sum)
            else:
                filtered.append(data[i])
        return filtered

    def callback(self, msg):
        ranges = [float('inf')] * self.num_ranges
        
        for point in msg.points:
            x, y, z = point.x, point.y, point.z

            if not (self.z_min <= z <= self.z_max):
                continue

            distance = math.hypot(x, y)
            if self.range_min <= distance <= self.range_max:
                angle = math.atan2(y, x)
                if self.angle_min <= angle <= self.angle_max:
                    index = int((angle - self.angle_min) / self.angle_increment)
                    if 0 <= index < self.num_ranges:
                        if distance < ranges[index]:
                            ranges[index] = distance

        ranges = self.gaussian_filter(ranges, window_size=5, sigma=1.0)

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
