import edu.wpi.first.vision.VisionPipeline;
import org.opencv.core.Mat;
import edu.wpi.first.networktables.NetworkTableInstance;
    
public class HubTracker implements VisionPipeline {
    public int val;
    private NetworkTableInstance nti;
       public HubTracker(NetworkTableInstance ntinst){
           nti = ntinst;
        }

    @Override
    public void process(Mat mat) {
      val += 1;
    }
  }

