import edu.wpi.first.vision.VisionPipeline;
import org.opencv.core.Mat;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.*;
import edu.wpi.cscore.CvSource;
    
public class CargoTracker implements VisionPipeline {
    public int val;
    private NetworkTableInstance nti;
    private NetworkTable cargoTable;
    private NetworkTableEntry cargoX;
    private NetworkTableEntry cargoY;
    private NetworkTableEntry cargoArea;
    private CvSource output;

    public CargoTracker(NetworkTableInstance ntinst, CvSource output_){
        nti = ntinst;
        cargoTable = nti.getTable("CARGO");
        cargoX = cargoTable.getEntry("X");
        cargoY = cargoTable.getEntry("Y");
        cargoArea = cargoTable.getEntry("Area");

        output = output_;
    }

    @Override
    public void process(Mat mat) {
      val += 1;
      output.putFrame(mat);
    }
  }

