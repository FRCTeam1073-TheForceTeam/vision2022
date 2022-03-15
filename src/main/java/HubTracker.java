import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.wpilibj.RobotBase;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.MatOfPoint;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.Imgcodecs;
import edu.wpi.cscore.CvSource;
    
public class HubTracker implements VisionPipeline {
    public int frameCounter;
    private NetworkTableInstance nti;
    private NetworkTable hubTable;
    private NetworkTableEntry hubX;
    private NetworkTableEntry hubY;
    private NetworkTableEntry hubArea;
    private NetworkTableEntry hubHMin;
    private NetworkTableEntry hubHMax;
    private NetworkTableEntry hubSMin;
    private NetworkTableEntry hubSMax;
    private NetworkTableEntry hubVMin;
    private NetworkTableEntry hubVMax;
    private NetworkTableEntry matchNuEntry;
    private NetworkTableEntry enabledState;
    private NetworkTableEntry saveHubImage;
    private CvSource output;
    private Mat hsvImage;
    private Mat maskImage;
    private Mat outputImage;
    private Mat erosionKernel;

    //private SimpleBlobDetectorParams blobParam

      public HubTracker(NetworkTableInstance ntinst, CvSource output_){
        nti = ntinst;
        hubTable = nti.getTable("HUB");
        hubX = hubTable.getEntry("Hub X");
        hubX.setDouble(0);
        hubY = hubTable.getEntry("Hub Y");
        hubY.setDouble(0);
        hubArea = hubTable.getEntry("Hub Area");
        hubArea.setDouble(0);
        hubHMin = hubTable.getEntry("H Min");
        hubHMin.setDouble(110);
        hubHMax = hubTable.getEntry("H Max");
        hubHMax.setDouble(140);
        hubSMin = hubTable.getEntry("S Min");
        hubSMin.setDouble(90);
        hubSMax = hubTable.getEntry("S Max");
        hubSMax.setDouble(255);
        hubVMin = hubTable.getEntry("V Min");
        hubVMin.setDouble(80);
        hubVMax = hubTable.getEntry("V Max");
        hubVMax.setDouble(255);
        saveHubImage = hubTable.getEntry("Save Hub Images");
        saveHubImage.setBoolean(false);

        matchNuEntry = nti.getTable("FMSInfo").getEntry("MatchNumber");

        output = output_;
        hsvImage = new Mat();
        maskImage = new Mat();
        outputImage = new Mat();
        erosionKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7,7));
     }

    @Override
    public void process(Mat inputImage) {
      frameCounter += 1;

      Imgproc.cvtColor(inputImage, hsvImage, Imgproc.COLOR_BGR2HSV_FULL);
      Core.inRange(hsvImage, new Scalar(hubHMin.getDouble(110), hubSMin.getDouble(90), hubVMin.getDouble(80)), new Scalar(hubHMax.getDouble(140), hubSMax.getDouble(255), hubVMax.getDouble(255)), maskImage);
     // outputImage.setTo(new Scalar(0,0,0));
      Imgproc.resize(inputImage, outputImage, new Size(inputImage.cols()/2, inputImage.rows()/2));
     // Core.bitwise_and(inputImage, inputImage, outputImage, maskImage);

      // Erode the mask image to eliminate the little "noise" pixels
      Imgproc.erode(maskImage, maskImage, erosionKernel);

      // Set up to find contours in the mask image.
      List<MatOfPoint> contours = new ArrayList<>();
      Mat hierarchy = new Mat();
      Imgproc.findContours(maskImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

      double totalArea = 0.0;
      double totalX = 0.0;
      double totalY = 0.0;
      double averageArea = 0.0;
      double averageX = 0.0;
      double averageY = 0.0;

      for (int cidx=0; cidx < contours.size(); cidx++){
        double area = Imgproc.boundingRect(contours.get(cidx)).area();
        if (area > 30){
        totalArea += area;
        }
      }

      // Walk the list of contours that we found and draw the ones that meet certain criteria:
      for (int cidx=0; cidx < contours.size(); cidx++) {
        // Grab the bounding rectangle for contour:
        Rect bounds = Imgproc.boundingRect(contours.get(cidx));
        Rect bounds2 = new Rect(bounds.x/2, bounds.y/2, bounds.width/2, bounds.height/2);
        double aspecterr = Math.abs(1.0 - (double)bounds.width/(double)bounds.height);
      //TODO:actually draw rectangle on outputImage  
        Imgproc.rectangle(outputImage, bounds2.br(), bounds2.tl(), new Scalar(0,0,255));

        if (bounds.area() > 30) {
          totalX += bounds.x * bounds.area();
          totalY += bounds.y * bounds.area();
          Imgproc.rectangle(outputImage, bounds2.br(), bounds2.tl(), new Scalar(0,0,255));
        }
      }

      if (contours.size() > 0 && totalArea > 0) {
        averageX = totalX/totalArea;
        averageY = totalY/totalArea;
        averageArea = totalArea/contours.size();
        hubX.setDouble(averageX);
        hubY.setDouble(averageY);
        hubArea.setDouble(averageArea);


        Imgproc.line(outputImage, new Point(averageX/2, 0), new Point(averageX/2, outputImage.rows()-1), new Scalar(0, 255, 0));
        Imgproc.line(outputImage, new Point(0, averageY/2), new Point(outputImage.cols()-1, averageY/2), new Scalar(0, 255, 0));
      }
      else {
        hubX.setDouble(0);
        hubY.setDouble(0);
        hubArea.setDouble(0);
      }

      // Line for 1 meter
      Imgproc.line(outputImage, new Point(0, 6/2), new Point(outputImage.cols()-1, 6/2), new Scalar(0, 0, 255));
      // Line for 2 meters
      Imgproc.line(outputImage, new Point(0, 191/2), new Point(outputImage.cols()-1, 191/2), new Scalar(0, 0, 255));
      // Line for 3 meters
      Imgproc.line(outputImage, new Point(0, 299/2), new Point(outputImage.cols()-1, 299/2), new Scalar(0, 0, 255));
      // Line for 4 meters
      Imgproc.line(outputImage, new Point(0, 372/2), new Point(outputImage.cols()-1, 372/2), new Scalar(0, 0, 255));
      // Line for 5 meters
      Imgproc.line(outputImage, new Point(0, 423/2), new Point(outputImage.cols()-1, 423/2), new Scalar(0, 0, 255));

     // Imgproc.Sobel(inputImage, outputImage, -1, 1, 1);
      //Imgproc.line(outputImage, new Point(0, outputImage.rows()/2), new Point(outputImage.cols()-1, outputImage.rows()/2), new Scalar(0, 0, 255));

      output.putFrame(outputImage);
     
    if (saveHubImage.getBoolean(false) == true) {
      //writes image files of what the hub tracker camera sees
     if (frameCounter%40 == 0){
      //Names files based on the match number from FMSInfo and frame number
      String fileName = String.format( "/media/usb_key/hub_match_%d_image_%d.jpg", matchNuEntry.getNumber(0).intValue(), frameCounter);
      if (Imgcodecs.imwrite(fileName, inputImage) == false){
      System.out.println("failed");
      }
      else {
        System.out.println("Success");
        }
      }
    }
  }
}
  

