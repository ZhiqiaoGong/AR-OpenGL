#include <opencv2/aruco.hpp> 
#include <opencv2/highgui.hpp>
using namespace cv;

int main()
{
	//create marker
	Mat markerImage;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
	aruco::drawMarker(dictionary, 0, 200, markerImage, 1);
	std::string out_file_name = "E:/marker.jpg"; //where to save the picture
	imwrite(out_file_name, markerImage);
	waitKey(0);
}
