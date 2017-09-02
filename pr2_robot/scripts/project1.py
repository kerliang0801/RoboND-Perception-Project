#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


#    VoxelGrid Downsampling Filter - lesson 3, sub 09, 10
#    ExtractIndices Filter
#    PassThrough Filter            - lesson 3, sub 11
#    RANSAC Plane Fitting          - lesson 3, sub 14
#    Outlier Removal Filter


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def outlier_filter(pcl_data):
    # start by creating a filter object to remove noise
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    # set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # any point with a mean distance larger than global (mean distance+x*std_dev)
    # will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(0.5)
    # finally call the filter function for magic
    pcl_objects = outlier_filter.filter()
    pcl.save(pcl_objects, 'cloud_filtered1.pcd')
    outlier_filter.set_negative(True)
    pcl.save(outlier_filter.filter(), 'cloud_noise1.pcd')
    return pcl_objects


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs:
    rospy.loginfo('Ex-2')
    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)  # XYZRGB
    #print('size1: ', pcl_data.size)
    pcl_data.to_file('pcl_data.pcd')
    pcl_data = outlier_filter(pcl_data)
    #print('size2: ', pcl_data.size)

    #cloud_filtered = pcl_data

    # TODO: Voxel Grid Downsampling
    vox = pcl_data.make_voxel_grid_filter()
    # try a voxel size and set to cloud
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # call filter function to obtain the resultant downsampled PC
    cloud_filtered = vox.filter()
    # save file
    ##filename = 'voxel_sampled1.pcd'
    ##pcl.save(cloud_filtered, filename)
    #print('size3: ', cloud_filtered.size)

    # TODO: PassThrough Filter to isolate the table and objects
    passthrough_z = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object
    filter_axis = "z"
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.6  # 0.6, 1.1
    axis_max = 0.85 # 1.5  1.1 
    passthrough_z.set_filter_limits(axis_min, axis_max)
    #passthrough.set_filter_limits(0, 1.5)
    # use filter function to obtain only table and objects
    cloud_filtered = passthrough_z.filter()
    passthrough_y = cloud_filtered.make_passthrough_filter()
    passthrough_y.set_filter_field_name("y")
    axis_min = -0.42
    axis_max = 0.42
    passthrough_y.set_filter_limits(axis_min, axis_max)
    # use filter function to obtain only table and objects
    cloud_filtered = passthrough_y.filter()
    # save result
    filename = 'pass_through_filtered1.pcd'
    pcl.save(cloud_filtered, filename)
    #print('size4: ', cloud_filtered.size)

    # TODO: RANSAC Plane Segmentation, will table and objects
    # seg = cloud_filtered.make_segmenter_normals(ksearch=50)
    seg = cloud_filtered.make_segmenter()
    seg.set_optimize_coefficients(True)
    # TODO: Extract inliers and outliers
    # set the model we wish to fit
    # seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # seg.set_max_iterations(100)
    max_distance = 0.018  # 0.01
    # set max distance for a point to be considered fitting the model
    seg.set_distance_threshold(max_distance)
    # call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    #print(coefficients)
    # extracted inliers - table
    pcl_table = cloud_filtered.extract(inliers, negative=False)
    # save file table
    filename = 'pcl_table1.pcd'
    pcl.save(pcl_table, filename)
    # extracted outliers - objects
    pcl_objects = cloud_filtered.extract(inliers, negative=True)
    # save file objects
    filename = 'pcl_objects1.pcd'
    pcl.save(pcl_objects, filename)

    # Outlier Removal Filter
    # pcl_objects = outlier_filter(pcl_objects)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(pcl_objects)
    tree = white_cloud.make_kdtree()
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.04)  # .08
    ec.set_MinClusterSize(30)
    ec.set_MaxClusterSize(100000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(pcl_objects)
    ros_cloud_table = pcl_to_ros(pcl_table)
    # now publish ros_cluster_cloud
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:
    rospy.loginfo('Ex-3')
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_list = []
    detected_labels = []

    for index, pts_list in enumerate(cluster_indices):
        features = []
        #print(index, len(cluster_indices))
        # Grab the points for the cluster
        pcl_cluster = pcl_objects.extract(pts_list)
        pcl_cluster.to_file('pcl_cluster1-{0}.pcd'.format(index))

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster_cloud = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster_cloud, using_hsv=True)
        normals = get_normals(ros_cluster_cloud)
        nhists = compute_normal_histograms(normals)
        features = np.concatenate((chists, nhists))
        ###labeled_features.append([feature, model_name])
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        sc = scaler.transform(features.reshape(1, -1))
        prediction = clf.predict(sc)
        label = encoder.inverse_transform(prediction)[0]
        detected_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects_list.append(do)

    # Objects detected
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_labels), detected_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# grab object info
def grab_obj_info(object_list):
    labels = []
    centroids = []
    for obj in object_list:
        labels.append(obj.label)
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(obj.cloud).to_array()
        # TODO: Create 'place_pose' for the object
        centroids.append(np.mean(points_arr, axis=0)[:3])
    return labels, centroids


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # TODO: Initialize variables
    labels = []
    centroids = []
    centroid = []
    object_list_param = []
    object_name = String()
    object_group = String()
    dict_list = []
    yaml_dict = []
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    labels, centroids = grab_obj_info(object_list)
    # TODO: Parse parameters into individual variables
    for i in range(0, len(object_list_param)):
        object_name.data = object_list_param[i]['name']
        object_group.data = object_list_param[i]['group']
        #print(i, object_name.data, object_group.data, centroids, labels)
        # check if is a valid param
        if object_name.data not in labels:
            rospy.loginfo("Object {0} not detected in scene!".format(object_name.data))
            continue
            # TODO: Rotate PR2 in place to capture side tables for the collision map

            # TODO: Loop through the pick list
        # correlate the name in list with object detected to find centroid
        label = None
        centroid = None
        for j in range(0, len(labels)):
            if object_name.data == labels[j]:
                label = labels[j]
                centroid = centroids[j]
                break

        # Initialize variables
        test_scene_num = Int32()
        arm_name = String()
        pick_pose = Pose()
        place_pose = Pose()

        # TODO: Assign the arm to be used for pick_place
        # green group(box): right arm, red group(box): left arm
        test_scene_num.data = 1
        pick_pose.position.x = float(centroid[0])
        pick_pose.position.y = float(centroid[1])
        pick_pose.position.z = float(centroid[2]) - 0.1

        dropbox_list_param = rospy.get_param('/dropbox')
        for j in range(0, len(dropbox_list_param)):
            if object_group.data == dropbox_list_param[j]['group']:
                arm_name.data = dropbox_list_param[j]['name']
                place_pose.position.x = float(dropbox_list_param[j]['position'][0])
                place_pose.position.y = float(dropbox_list_param[j]['position'][1])
                place_pose.position.z = float(dropbox_list_param[j]['position'][2])
                break
        print(object_name)
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ", resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # TODO: Output your request parameters into output yaml file
    yaml_filename = 'object_list.yaml'
    send_to_yaml("object_{0}.yaml".format(test_scene_num.data), dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=10)
    # TODO: Create Publishers
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
