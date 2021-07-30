#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>

#include <cmath>

using namespace std;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr  myCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

void assign_part_id(pcl::PointCloud<pcl::PointXYZRGB> &cloud, int i)
{
  for(int k=0; k < cloud.size(); k++)
  {
    //cloud.points[k].r = 0; cloud.points[k].g = 0; cloud.points[k].b = i+1; // id: 1,2,3,4... 
    cloud.points[k].r = 255*i; cloud.points[k].g = 100*(i+1); cloud.points[k].b = i+1; // id: 1,2,3,4... 
  
  }
}

void loadPointCloud(std::string data_dir, std::string object_name)
{
  std::string data_path = data_dir + object_name + ".ply";
  pcl::io::loadPLYFile<pcl::PointXYZRGB> (data_path, *cloud);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_color");
  std::cerr << "\n"<< "---------------------add color to point cloud---------------------" << "\n";
  ros::NodeHandle nh_, cloud_n, obb_n;
  ros::Publisher cloud_pub = cloud_n.advertise<sensor_msgs::PointCloud2> ("myCloud", 1);
  ros::Rate loop_rate(10);

  std:string data_dir;
  std::vector<std::string> objects_name;
  bool load_background, color_segment;

  nh_ = ros::NodeHandle("~");
  nh_.getParam("data_dir", data_dir);
  nh_.getParam("objects_name", objects_name);

  for(int i=0; i < objects_name.size(); ++i)
  {
    loadPointCloud(data_dir, objects_name[i]);
    assign_part_id(*cloud, i);
    *myCloud += *cloud;
  }

  std::string saved_path = data_dir + "parts_with_id.ply";
  pcl::io::savePLYFileBinary (saved_path, *myCloud);


  pcl::PCLPointCloud2 cloud_filtered;
  sensor_msgs::PointCloud2 output;
  myCloud->header.frame_id = "camera_depth_optical_frame";
  pcl::toPCLPointCloud2(*myCloud, cloud_filtered);
  pcl_conversions::fromPCL(cloud_filtered, output);
  
  while (ros::ok())
  {  
    cloud_pub.publish (output);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}