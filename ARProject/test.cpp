#include <opencv2/aruco.hpp> 
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	//���ɱ��
	Mat markerImage;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
	aruco::drawMarker(dictionary, 1, 200, markerImage, 1);

	//����
	string out_file_name = "E:/gzq2/ѧϰ/������/��ǿ��ʵ/ʵ��/marker_6x6_1.jpg";
	imwrite(out_file_name, markerImage);
	waitKey(0);

}
