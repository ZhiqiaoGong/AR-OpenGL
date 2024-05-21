#include <opencv2/aruco.hpp> 
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	//生成标记
	Mat markerImage;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
	aruco::drawMarker(dictionary, 1, 200, markerImage, 1);

	//保存
	string out_file_name = "E:/gzq2/学习/大三下/增强现实/实验/marker_6x6_1.jpg";
	imwrite(out_file_name, markerImage);
	waitKey(0);

}
