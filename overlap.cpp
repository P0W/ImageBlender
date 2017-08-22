// Blending images of different sizes, overlaying on video

#include <chrono>
#include <iomanip> 

#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

namespace 
{
   std::string current_time_and_date()
   {
      auto now = std::chrono::system_clock::now();
      auto in_time_t = std::chrono::system_clock::to_time_t(now);

      std::stringstream ss;
      ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
      return ss.str();
   }

   void displayText( cv::Mat& img, const std::string& text  )
   {
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 1;
      int thickness = 1;
      int baseline=0;

      cv::Size textSize = cv::getTextSize(text , fontFace,
         fontScale, thickness, &baseline);
      baseline += thickness;

      // center the text
      cv::Point textOrg((img.cols - textSize.width)/2,
         (img.rows + textSize.height)/2);

      // draw the box
      cv::rectangle(img, textOrg + cv::Point(0, baseline),
         textOrg + cv::Point(textSize.width, -textSize.height),
         cv::Scalar(0,0,255));

      // ... and the baseline first
      cv::line(img, textOrg + cv::Point(0, thickness),
         textOrg + cv::Point(textSize.width, thickness),
         cv::Scalar(0, 0, 255));

      // then put the text itself
      cv::putText(img, text, textOrg, fontFace, fontScale,
         cv::Scalar(0, 255, 0), thickness, cv::LINE_AA );
   }

   void blendImage( cv::Mat& srcImg, const cv::Mat& overlayImg, 
      cv::Rect* roi = NULL, double opacity = 0.2, int lastCols = 0, int lastRows = 0 )
   {
      cv::Rect calRoi;
      if( roi == NULL )
      {
         calRoi = cv::Rect(
            srcImg.cols - overlayImg.cols - lastCols, 
            srcImg.rows - overlayImg.rows - lastRows,
            overlayImg.cols, 
            overlayImg.rows
            );
      }
      else
      {
         calRoi = *roi;
      }

      cv::addWeighted( srcImg(calRoi), opacity, overlayImg, 1-opacity, 0.0, srcImg(calRoi) );
   }

   /**
   * @brief Calculates the parametric points on two ellipse , give by
   * ( acos(theta), bsin(theta) ) and ( (a+15)cos(theta), (b+15)sin(theta) )
   */
   std::pair<cv::Point, cv::Point> getTick( double theta, int a, int b, const cv::Point& origin )
   {
      // parameterized equation for ellipse
      return std::make_pair<cv::Point, cv::Point>( 
         cv::Point( static_cast<int>(origin.x+a*cos( theta )), static_cast<int>(origin.y+b*sin( theta ))), 
         cv::Point( static_cast<int>(origin.x+(a+15)*cos( theta )) , static_cast<int>(origin.y+(b+15)*sin( theta ))) ) ;
   }

   /**
   * @brief Embeds a dial 0-120 degree in the given image

   * @param x [in] the x-radius of ellipse
   * @param y [in] the y-radius of ellipse
   * @return none
   */
   void embedDial( cv::Mat& dst, const cv::Point& origin, int x, int y )
   {
      /* http://www.tydac.ch/color/ */
      auto color = cv::Scalar( 110, 117, 63 ) ;

      for( int32_t theta = 0 ; theta <= 120; theta += 10 )
      {

         auto p = getTick( -theta*CV_PI/180.0, x, y, origin );

         cv::line( dst, p.first, p.second, color, 2, 32 );

         if( theta % 30 == 0 )
         {
            cv::putText( dst, std::to_string( theta ), 
               cv::Point( 
               static_cast<int>(origin.x + (x+20)*cos(-theta*CV_PI/180.0 )), 
               static_cast<int>(origin.y + (y+20)*sin(-theta*CV_PI/180.0 ))),
               1, 1, color, 1, cv::LINE_4 );
         }

         /* hard coded , for now */
         if( theta == 60 )
         {
            cv::line( dst, origin, p.first, color, 2, cv::LINE_8  );
         }
      }

   }

}

int main( )
{

   cv::VideoCapture cap(0); // open the default camera

   if(!cap.isOpened())
   {
      return -1;
   }

   cv::namedWindow( "Blended -P0W!", 1 );

   // This is annoying I want it here :-|
   cv::moveWindow( "Blended -P0W!", -900, 300 );

   /// Establish the number of bins
   int histSize = 256;

   /// Set the ranges ( for B,G,R) )
   float range[] = { 0, 256 } ;
   const float* histRange = { range };

   cv::Mat b_hist, g_hist, r_hist, mask;
   cv::Mat src, dst ;
   std::vector<cv::Mat> bgr_planes;

   // Draw the histograms for B, G and R
   int hist_w = 200 ; 
   int hist_h = 50;

   int bin_w = cvRound( (double) hist_w/histSize );

   cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

   // Place image at bottom right corner
   cv::Rect roi = cv::Rect(
      (int)cap.get(CV_CAP_PROP_FRAME_WIDTH) - hist_w, 
      (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) - hist_h,
      histImage.cols, 
      histImage.rows
      );


   cv::Mat pfd =  cv::imread( "C:/opencv-3.3.0/camera_test/pfd.bmp" ); 
   cv::Mat cargobay = cv::imread( "C:/opencv-3.3.0/camera_test/missionplan.png" );
   if( pfd.data == NULL || cargobay.data == NULL )
   {
      return -1;
   }

   do
   {
      cap >> src ;

      cv::flip(src, dst, 1); // Flip the image
      src.release();

      cv::resize( dst, dst, cv::Size(800, 600) );

      cv::split( dst, bgr_planes );

      /// Compute the histograms:
      cv::calcHist( &bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange );
      cv::calcHist( &bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange );
      cv::calcHist( &bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange );

      /// Normalize the result to [ 0, histImage.rows ]
      cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, mask );
      cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, mask );
      cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, mask );

      histImage.setTo( cv::Scalar( 0,0,0) );

      /// Draw for each channel
      for( int i = 1; i < histSize; i++ )
      {
         cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
            cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
            cv::Scalar( 255, 0, 0), 2, 8, 0  );

         cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
            cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
            cv::Scalar( 0, 255, 0), 2, 8, 0  );

         cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
            cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
            cv::Scalar( 0, 0, 255), 2, 8, 0  );
      }

      displayText( histImage, current_time_and_date() );


      blendImage( dst, cargobay , NULL, 0.6 );
      blendImage( dst, pfd, NULL, 0.5, 0, cargobay.rows );
      blendImage( dst, histImage, NULL, 0.6, 
         pfd.cols  , 
         cargobay.rows  );

      embedDial( dst, cv::Point(  100, dst.rows - 90  ), 100, 90  );

      cv::imshow( "Blended -P0W!", dst );

   }while(cv::waitKey(1) != 27) ; // Wait for ESC key

   cv::destroyAllWindows();
   cap.release();
}
