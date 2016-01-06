// 
// サンプルコードとして，ブロックマッチングを用いたロゴの識別法を示します．
// 主な目的は，読み込んだアノテーション情報の取得方法や，画像の読み込み方，
// 認識結果の登録方法を示すことです．
//
// 情報を管理する配列には C++ 言語の std::list と std::vector が用いられており，
// こちらで用意した関数を用いずに，直接操作していただいても構いません．
//
// Here, we show a classification method by block matching as a sample code.
// The main purpose of us is to show how to access to the annotation information,
// read images, and append the classification results to the output lists.
//
// As arrays to manage the information, std::list and std::vector of C++ are used.
// One can use the member functions of them directly without using our prepared functions.
//

#include "prmu.hpp"

// include OpenCV libs
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

// for the MSVC
#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif

#pragma comment(lib, "opencv_core"	CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_highgui"	CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_imgproc"	CV_VERSION_STR CV_EXT_STR)



//
// ---------------------------------------------------------------------------------
//

using namespace std;
using namespace cv;

namespace sample{
  void dft_based_ncc( Mat& C, Mat& I, Mat& K );
  void psf2otf( Mat& dst, const Mat& src, size_t rows, size_t cols );
  void circshift( Mat& dst, const Mat& src, int dx, int dy );
}

//
// ---------------------------------------------------------------------------------
//

Mat damy;

int fs;

//
//
//

class Learn_
{
public:
  Mat img[150];
  string Label; // 画像に対応するラベル
  string Fpath; // 画像ファイルへのパス
  int lwc;
  
  Learn_(string lab)
  {
    Label = lab;
  }
  
public:
  void setPath()
  {
    img[0] = damy;
  }

  string getLabel()
  {
    return Label;
  }
  
  string getPath()
  {
    return Fpath;
  }
  
};

int counta =0;
//
//
//

vector<Learn_> store_learn();
int recognize(vector<Learn_> le, Mat rect);
//
//
//
int FILA_j;


void user_function(
		   //
		   // output
		   prmu::ImageList (&imlist_result)[3], // 結果情報の記録用
		   //
		   // input
		   size_t lv,
		   const prmu::ImageList& imlist_learn,
		   const prmu::ImageList (&imlist_test)[3]
		   )
{
  
  prmu::ImageList::const_iterator ite_learn, ite_test;
  prmu::ImageList::iterator ite_result;
  
   
  // 学習用画像の格納
  vector<Learn_> learn_list = store_learn();


    //評価用画像と特徴量を比較するテンプレート画像の作成
  vector<cv::Mat> src_img;

  
  ite_learn = imlist_learn.begin();
  
  for(int i = 0;i<8;i++){
    ++ite_learn;
  }
  learn_list[0].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi1( ite_learn->rect_of_1st_img() );
  learn_list[0].img[0] = learn_list[0].img[0]( Rect(_roi1.x+22, _roi1.y+12, _roi1.w-50, _roi1.h-26) );
  resize(learn_list[0].img[0], learn_list[0].img[0], Size(101,101));

  for(int i = 0;i<21-8;i++){
    ++ite_learn;
  }
  learn_list[1].img[0]= imread(ite_learn->full_file_path());
  prmu::Rect _roi2( ite_learn->rect_of_1st_img() );
  learn_list[1].img[0] = learn_list[1].img[0](Rect(_roi2.x-6, _roi2.y-3, _roi2.w+8, _roi2.h+6));
  resize(learn_list[1].img[0],learn_list[1].img[0], Size(101,101));

  for(int i = 0;i<22;i++){
    ++ite_learn;
  }
  learn_list[2].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi3( ite_learn->rect_of_1st_img() );
  learn_list[2].img[0] = learn_list[2].img[0]( Rect(_roi3.x-2, _roi3.y, _roi3.w+3, _roi3.h) );
  resize(learn_list[2].img[0], learn_list[2].img[0], Size(101,101));
  //resize(FILA_0001, FILA_0001, Size(180,180));

  
  learn_list[12].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi13( ite_learn->rect_of_1st_img() );
  learn_list[12].img[0] = learn_list[12].img[0]( Rect(_roi13.x-1, _roi13.y+22, _roi13.w-5, _roi13.h-22) );
  resize(learn_list[12].img[0], learn_list[12].img[0], Size(101,101));
  //resize(FILA_0001, FILA_0001, Size(180,180));

  for(int i = 0;i<17;i++){
    ++ite_learn;
    
  }
  learn_list[3].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi11( ite_learn->rect_of_1st_img() );
  learn_list[3].img[0] = learn_list[3].img[0]( Rect(_roi11.x-10, _roi11.y-12, _roi11.w+19, _roi11.h+21) );
  resize(learn_list[3].img[0], learn_list[3].img[0], Size(101,101));
  medianBlur(learn_list[3].img[0], learn_list[3].img[0], 5);
  learn_list[3].img[0] =~ learn_list[3].img[0];

  for(int i = 0;i<13;i++){
    ++ite_learn;
  }
  learn_list[4].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi4( ite_learn->rect_of_1st_img() );
  learn_list[4].img[0] = learn_list[4].img[0]( Rect(_roi4.x-5, _roi4.y, _roi4.w+5, _roi4.h) );
  resize( learn_list[4].img[0], learn_list[4].img[0], Size(101,101));
  //resize( LECOQ_0007,  LECOQ_0007, Size(180,180));
  
  for(int i = 0;i<17;i++){
    ++ite_learn;
  }
  learn_list[5].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi12( ite_learn->rect_of_1st_img() );
  learn_list[5].img[0] = learn_list[5].img[0]( Rect(_roi12.x-4, _roi12.y, _roi12.w+10, _roi12.h) );
  resize(learn_list[5].img[0], learn_list[5].img[0], Size(101,101));  //resize(MIZUNO_m_0003,MIZUNO_m_0003 , Size(180,180));

  for(int i = 0;i<14;i++){
    ++ite_learn;
  }
  learn_list[6].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi5( ite_learn->rect_of_1st_img() );
  learn_list[6].img[0] = learn_list[6].img[0]( Rect(_roi5.x-4, _roi5.y-5, _roi5.w+4, _roi5.h+7) );
  resize(learn_list[6].img[0], learn_list[6].img[0], Size(101,101));
  //resize(MIZUNO_m_0003,MIZUNO_m_0003 , Size(180,180));

  for(int i = 0;i<7+3;i++){
    ++ite_learn;
  }
  learn_list[7].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi6( ite_learn->rect_of_1st_img() );
  learn_list[7].img[0] = learn_list[7].img[0]( Rect(_roi6.x+2, _roi6.y, _roi6.w, _roi6.h) );
  resize(learn_list[7].img[0], learn_list[7].img[0], Size(101,101));
  medianBlur(learn_list[7].img[0], learn_list[7].img[0],5);
  //resize(NB_0001, NB_0001, Size(180,180));

  for(int i = 0;i<14-3;i++){
    ++ite_learn;
  }
  learn_list[8].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi7( ite_learn->rect_of_1st_img() );
  learn_list[8].img[0] = learn_list[8].img[0]( Rect(_roi7.x-4, _roi7.y+1, _roi7.w+7, _roi7.h-2) );
  resize( learn_list[8].img[0], learn_list[8].img[0], Size(101,101));
  medianBlur(learn_list[8].img[0], learn_list[8].img[0],5);
  //resize( NB_n_0003,  NB_n_0003, Size(180,180));

  for(int i = 0;i<18;i++){
    ++ite_learn;
  }
  learn_list[9].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi8( ite_learn->rect_of_1st_img() );
  learn_list[9].img[0] = learn_list[9].img[0]( Rect(_roi8.x-20, _roi8.y-7, _roi8.w+38, _roi8.h+12) );
  //imshow("a",learn_list[9].img[0]);
  //waitKey(0);
  resize(learn_list[9].img[0], learn_list[9].img[0], Size(101,101));
   medianBlur(learn_list[9].img[0], learn_list[9].img[0],3);


  for(int i = 0;i<12;i++){
    ++ite_learn;
  }
  learn_list[10].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi9( ite_learn->rect_of_1st_img() );
  learn_list[10].img[0] = learn_list[10].img[0]( Rect(_roi9.x-3, _roi9.y, _roi9.w+5, _roi9.h) );
  resize( learn_list[10].img[0], learn_list[10].img[0], Size(101,101));
  //resize(UA_0002 , UA_0002 , Size(180,180));

  for(int i = 0;i<19;i++){
    ++ite_learn;
  }
  learn_list[11].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi10( ite_learn->rect_of_1st_img() );
  learn_list[11].img[0] = learn_list[11].img[0]( Rect(_roi10.x-6, _roi10.y-6, _roi10.w+10, _roi10.h+8) );
  resize(learn_list[11].img[0], learn_list[11].img[0], Size(101,101));
  //resize(YONEX_0007, YONEX_0007, Size(180,180));

  learn_list[13].img[0] = imread(ite_learn->full_file_path());
  prmu::Rect _roi14( ite_learn->rect_of_1st_img() );
  learn_list[13].img[0] = learn_list[13].img[0]( Rect(_roi14.x+8, _roi14.y-4, _roi14.w-2, _roi14.h-25) );
  resize(learn_list[13].img[0], learn_list[13].img[0], Size(101,101));

    
  // 学習用画像の処理
  for(int i=0; i<learn_list.size(); i++){
    //imshow("learn",learn_list[i].img[0]);
    //waitKey(0);
    cvtColor(learn_list[i].img[0], learn_list[i].img[0], CV_BGR2GRAY);
    threshold(learn_list[i].img[0], learn_list[i].img[0], 0, 255, THRESH_BINARY|THRESH_OTSU);
    rectangle(learn_list[9].img[0], Point(0,78), Point(22,100), cv::Scalar(0,0,0),-1,CV_AA);
	
	
    if(learn_list[i].Label == "LECOQ" || 
       learn_list[i].Label == "NB" ||
       learn_list[i].Label == "SHUN"
       ){
      learn_list[i].img[0] =~ learn_list[i].img[0];
    }
    //resize(learn_list[i].img[0], learn_list[i].img[0], Size(101,101), 0, 0, INTER_LANCZOS4);
    learn_list[i].img[0]=~ learn_list[i].img[0];

    // 回転： -90 [deg],  スケーリング： 1.0 [倍]
    float angle = -90.0, scale = 1.0;
    // 中心：画像中心
    cv::Point2f center(50, 50);
    // 以上の条件から2次元の回転行列を計算
    const cv::Mat affine_matrix = cv::getRotationMatrix2D( center, angle, scale );

    cv::warpAffine(learn_list[i].img[0], learn_list[i].img[1], affine_matrix, learn_list[i].img[0].size());
    cv::warpAffine(learn_list[i].img[1], learn_list[i].img[2], affine_matrix, learn_list[i].img[0].size());
    cv::warpAffine(learn_list[i].img[2], learn_list[i].img[3], affine_matrix, learn_list[i].img[0].size());

    
    for(int rot=0;rot<4;rot++){

      flip(learn_list[i].img[rot], learn_list[i].img[rot+4], 1);
      
    }

    for(int rot=0;rot<8;rot++){
      learn_list[i].img[rot+8] =~ learn_list[i].img[rot];
    }
    learn_list[i].lwc=0;
    for(int y=0; y<101; y++){
      for(int x=0; x<101; x++){
	if(static_cast<int>(learn_list[i].img[0].at<unsigned char>(y,x))==255){
	  learn_list[i].lwc++;
	}
      }
    }
    /*
    for(int tes=0;tes<16;tes++){
      stringstream sss;
      sss << tes;
      imwrite("./image/r" + learn_list[1].Label + sss.str() + ".jpg",learn_list[1].img[tes]);
    }
    */
  }
  int ite=0;
  for ( size_t _lv = 0; _lv < lv; ++_lv )
    {
      // アノテーション情報へのポインタ（イテレータ）
      // The pointer (actually iterator) of annotation information
      ite_test = imlist_test[_lv].begin();     // 入力用
      ite_result = imlist_result[_lv].begin(); // 結果用

      // アノテーション情報にアクセスするには，(*ite_test).XXX や ite_test->XXX が使用できます．
      // The access to the annotation information (i.e., member variables and functions)
      // is performed by (*ite_test).XXX or ite_test->XXX


      Rect box;
      for ( ; ite_test != imlist_test[_lv].end(); ++ite_test, ++ite_result ) // each test image
	{

	  ite_learn = imlist_learn.begin();
	  //
	  // ------  ------
	  //
		  
	  Mat input_img = imread(ite_test->full_file_path());
          ////////////////ここから//////////////
	  fs=0;
	  Mat src_img = input_img;
	  Mat src_img_cp = src_img.clone();
	  Mat src_img2 = src_img.clone();
	  
	    const int cluster_count = 24;

	  // 画像を1列の行列に変形
	  cv::Mat points;
	  src_img2.convertTo(points, CV_32FC3);
	  points = points.reshape(3, src_img2.rows*src_img2.cols);

	  // RGB空間でk-meansを実行
	  cv::Mat_<int> clusters(points.size(), CV_32SC1);
	  cv::Mat centers;
	  // クラスタ対象，クラスタ数，（出力）クラスタインデックス，
	  // 停止基準，k-meansの実行回数，手法，（出力）クラスタ中心値

	  cv::kmeans(points, cluster_count, clusters, 
		     cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);
        
	  // すべてのピクセル値をクラスタ中心値で置き換え
	  cv::Mat dst_img(src_img2.size(), src_img2.type());
	  cv::MatIterator_<cv::Vec3b> itd = dst_img.begin<cv::Vec3b>(), 
	    itd_end = dst_img.end<cv::Vec3b>();
	  for(int i=0; itd != itd_end; ++itd, ++i) {
	    cv::Vec3f &color = centers.at<cv::Vec3f>(clusters(i), 0);
	    (*itd)[0] = cv::saturate_cast<uchar>(color[0]);
	    (*itd)[1] = cv::saturate_cast<uchar>(color[1]);
	    (*itd)[2] = cv::saturate_cast<uchar>(color[2]);
	  }
	  /*
	  if(ite==6){
	  imshow("dst_img",dst_img);
	  waitKey(0);
	  }
	  */
	  string labname;
	  vector <Mat> color;
	  split(dst_img,color);
	  // src_imgをグレースケール化しgray_imgに代入、gray_imgを2値化しbin_imgに代入
	  cv::Mat gray_img, bin_img,h_img,m_img;
	  //cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);
		  
	  for(int i = 0;i<3;i++){
	    gray_img = color[i];
	    
	    if(ite==6&&i==2){
	      imwrite("./image/gray_img2.jpg",gray_img);
	    }
	    
	    Mat c_gray_img;
	    cvtColor(gray_img,c_gray_img,CV_GRAY2RGB);
	    
	      Mat canny_img;
	      Canny(gray_img, canny_img, 50, 200);
		      
	      //imshow("canny",canny_img);
	      //waitKey(0);
	      dilate(canny_img,canny_img,Mat(),Point(-1,-1),15);
	      erode(canny_img,canny_img,Mat(),Point(-1,-1),10);
	      //imshow("canny",canny_img);
	      //waitKey(0);
	      int mean,mnum;
	      mean=0;
	      mnum=0;
	      for(int i=0;i<canny_img.rows;i++){
		for(int j=0;j<canny_img.cols;j++){
		  if(static_cast<int>(canny_img.at<unsigned char>(i,j))!=0){
		    //gray_img.at<unsigned char>(i,j)=0;
		    // }
		    //else{
		    mean+=gray_img.at<unsigned char>(i,j);
		    mnum++;
		  }
		}
	      }
	      mean/=mnum;
		      
	      for(int i=0;i<canny_img.rows;i++){
		for(int j=0;j<canny_img.cols;j++){
		  if(static_cast<int>(canny_img.at<unsigned char>(i,j))==0){
		    //gray_img.at<unsigned char>(i,j)=mean;
		  }
		}
	      }
	      /*
	      if(ite==6){
		imwrite("./image/canny_def_img.jpg",gray_img);
	      }
	      */
	      //imshow("gray",gray_img);
	      //waitKey(0);
	      equalizeHist( gray_img, h_img );
	      medianBlur(h_img, m_img, 3);
	      adaptiveThreshold(m_img, bin_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 0);//51,2
	      erode(bin_img,bin_img,Mat(),Point(-1,-1),1);
	      dilate(bin_img,bin_img,Mat(),Point(-1,-1),1);

	      /*
	      if(ite==6){
		imwrite("./image/bin_img.jpg",bin_img);
	      }
	      */
		      
	      // 輪郭を管理する変数contours
	      std::vector<std::vector<cv::Point> > contours;

	      // 2値画像bin_imgで輪郭走査を行い、結果(各輪郭の点列)をcontoursに代入
	      cv::findContours(bin_img, contours, CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
	      int c,c2,n;
	      double v,v2;
	      // 各輪郭ごとに矩形生成を行う
	      for(int i=0; i<contours.size(); ++i){
		// 使用しない矩形を選択する(輪郭の長さでフィルタリング)
		size_t count = contours[i].size();
		if(count < 70 || count > 4000) continue;
		// 輪郭の点列を行列型に変換
		cv::Mat pointsf;
		cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
		// 輪郭を構成する点列をすべて包含する最小の矩形を計算
		box = cv::boundingRect(pointsf);
		n=0;
		c=0;
		v=0.0;
		c2=0;
		v2=0.0;

		rectangle(src_img_cp, box.tl(), box.br(), cv::Scalar(255,255,255),2,CV_AA);

		Mat crop_img(m_img,Rect(box.tl().x,box.tl().y,box.width,box.height));
		resize(crop_img, crop_img, Size(100,100), 0, 0, INTER_LANCZOS4);
		Mat bin_crop;
		threshold(crop_img, bin_crop, 0, 255, THRESH_BINARY|THRESH_OTSU);
		/*
		if(ite==6){
		  imshow("c",crop_img);
		  waitKey(0);
		}
		*/
		for(int k=0; k<100; k+=1){
		  for(int j=0; j<100; j+=1){
		    c+=crop_img.at<unsigned char>(k,j);
		    c2+=static_cast<int>(bin_crop.at<unsigned char>(j,k));
		    n++;
		  }
		}
		c/=n;
		c2/=n;
		for(int k=0; k<100; k+=1){
		  for(int j=0; j<100; j+=1){
		    v += (crop_img.at<unsigned char>(k,j) - c) * (crop_img.at<unsigned char>(k,j) - c);
		    v2 += (static_cast<int>(bin_crop.at<unsigned char>(j,k)) - c2) *(static_cast<int>(bin_crop.at<unsigned char>(j,k)) - c2);
		  }
		}
		v /= n;
		v2/=n;
		//cout << v << endl;
		if(v<5000){
		  //imshow("c",crop_img);
		  // waitKey(0);
		  continue;
		}
		//cout<<v2<<endl;
		if(v2<10000){
		  //imshow("b",bin_crop);
		  //waitKey(100);
		  continue;
		}
		
		//rectangle(src_img_cp, box.tl(), box.br(), cv::Scalar(255,255,255),2,CV_AA);
		/////////////以下認識//////////////
		  int recnum;
		  Mat roi_img(c_gray_img, box);
		  
		  recnum = recognize(learn_list, roi_img);
			
		  if(recnum != -1){
		    labname = learn_list[recnum].Label;
		    prmu::Rect bbox;
		    if(recnum == 12){
		      if((FILA_j+4)%4==0){
			bbox.x = box.x;
			bbox.y = box.y-box.height*0.5;
			bbox.w = box.width * 1.1;
			bbox.h = box.height * 1.5;
		      }
		      else{
			bbox.x = box.x; // 矩形の左上の座標 (x,y) 横幅と縦幅 (w,h)
			bbox.y = box.y;
			bbox.w = box.width; 
			bbox.h = box.height;
		      }
		    }
		    else if(recnum == 13){
			bbox.x = box.x - box.width * 0.15;
			bbox.y = box.y;
			bbox.w = box.width * 1.2;
			bbox.h = box.height * 1.5;
		    }
		    else if(recnum == 0){
			bbox.x = box.x - box.width * 0.6;
			bbox.y = box.y - box.height * 0.5;
			bbox.w = box.width * 2.2;
			bbox.h = box.height * 2.2;
		    }
		    else if(recnum == 9){
			bbox.x = box.x + box.width * 0.15;
			bbox.y = box.y + box.height * 0.1;
			bbox.w = box.width * 0.7;
			bbox.h = box.height * 0.8;
			//imshow("test",test_img);
			//waitKey(0);
		    }
		    else {
		    bbox.x = box.x; // 矩形の左上の座標 (x,y) 横幅と縦幅 (w,h)
		    bbox.y = box.y;
		    bbox.w = box.width; 
		    bbox.h = box.height;
		    }
		    //int face[] = {cv::FONT_HERSHEY_SIMPLEX, cv::FONT_HERSHEY_PLAIN, cv::FONT_HERSHEY_DUPLEX, cv::FONT_HERSHEY_COMPLEX,cv::FONT_HERSHEY_TRIPLEX, cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SCRIPT_SIMPLEX,cv::FONT_HERSHEY_SCRIPT_COMPLEX, cv::FONT_ITALIC};
		    rectangle(src_img_cp, Point(bbox.x,bbox.y), Point(bbox.x+bbox.w,bbox.y+bbox.h), cv::Scalar(0,0,255),2,CV_AA);
		    //cv::putText(src_img_cp, labname, cv::Point(bbox.x,bbox.y), face[0], 1, cv::Scalar(255,255,255), 2, CV_AA);
		    /////////////////////////結果格納///////////////////
		    ite_result->append_result( prmu::label::str2label( labname ) , bbox );
		    ///////////////////////////////////////////////////
		  }
	      }
	  }
	  ite++;
	  if(ite==15){
	    imshow("src",src_img_cp);
	    waitKey(0);
	    }
	  //cout << counta++ << endl;
	}
    }
}

vector<Learn_> store_learn()
{
  vector<Learn_> llist;

  Learn_ l1("ASICS");
  Learn_ l2("ASICS");
  Learn_ l3("FILA");
  Learn_ l4("FILA");
  Learn_ l5("LECOQ");
  Learn_ l6("MIZUNO");
  Learn_ l7("MIZUNO");
  Learn_ l8("NB");
  Learn_ l9("NB");
  Learn_ l10("SHUN");
  Learn_ l11("UA");
  Learn_ l12("YONEX");
  Learn_ l13("FILA");
  Learn_ l14("YONEX");

  
  llist.push_back(l1);
  llist.push_back(l2);
  llist.push_back(l3);
  llist.push_back(l4);
  llist.push_back(l5);
  llist.push_back(l6);
  llist.push_back(l7);
  llist.push_back(l8);
  llist.push_back(l9);
  llist.push_back(l10);
  llist.push_back(l11);
  llist.push_back(l12);
  llist.push_back(l13);
  llist.push_back(l14);

  
  for(int i=0; i<14; i++)
    llist[i].setPath();
  
  
  return llist;
}



/*------------------------------------------------
  / 認識
  / -----------------------------------------------*/
int recognize(vector<Learn_> le, Mat rect)
{
  
  if((double)rect.rows / rect.cols > 5 ||
     (double)rect.cols / rect.rows > 7 ||
     rect.cols <= 30 ||
     rect.rows <= 10 ||
     rect.rows >400  ||
     rect.cols >600
     )
    return -1;
  
  // imshow("ORG_RECT", rect);
  
  int num=-1;
  double MAXV=0;
  int fj;
  cvtColor(rect, rect, CV_BGR2GRAY);
  medianBlur(rect, rect, 5);
  threshold(rect, rect, 0, 255, THRESH_BINARY|THRESH_OTSU);
  resize(rect, rect, Size(101,101), 0, 0, INTER_LANCZOS4);
  
  //imshow("Rect", rect);
  //waitKey(0);
  
  int wc=0;
  for(int y=0; y<101; y++){
    for(int x=0; x<101; x++){
      if(static_cast<int>(rect.at<unsigned char>(y,x))==255){
	wc++;
      }
    }
  }
  for(int i=0; i<le.size(); i++){
    if( (wc>le[i].lwc*0.95) && (wc<le[i].lwc*1.05) ){
      //cout << "continue " << le[i].Label; 
      continue;
    }
    if(fs==1&&i==12){
      continue;
    }
    for(int j=0; j<16; j++){
      if(wc < 3000 && j > 7){
	continue;
      }
      if(wc > 7200 && j < 8){
	continue;
      }
      Mat tmp = le[i].img[j].clone();
      Mat A;
      double maxv=0;
      Point maxp;
      double th=0.66;
      if(i==12){
	th=0.8;
      }
      if(i==11||i==5||i==13||i==7||i==14||i==9){
	th=0.604;
      }
      if(i==6||i==7||i==8){
	th=0.5;
      }
      if(i==15){
	th=7.0;
      }

      matchTemplate(rect, tmp, A, CV_TM_CCOEFF_NORMED);
      minMaxLoc(A, NULL, &maxv, NULL, &maxp);

      //imshow("tmp",tmp);
      //waitKey(100);
    
      if(MAXV < maxv){
	MAXV = maxv;
      }

      if(maxv >= th){
	num = i;
	//cout << le[i].Label << i << endl;
	//cout << maxv << endl;
	//imshow("rect", rect);
	//imshow("TMP", tmp);
	//waitKey(100);
	if(i!=12){
	  if(maxv > 0.75){
	    fs=1;
	    break;
	  }
	}
	else{
	  fj = j;
	}
      }
      
    }
  }
  if(num==12){
    FILA_j=fj;
  }
  return num;
}
