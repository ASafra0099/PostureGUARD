import * as React from "react";
import { StyleSheet,Image, View, TouchableOpacity, Text } from "react-native";
import { Svg, Circle, Path } from "react-native-svg";
import { LinearGradient } from "expo-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";
import { useNavigation, ParamListBase } from "@react-navigation/native";
import { Border, FontSize, FontFamily, Color } from "../GlobalStyles";
import GoBackButton from "../Components/goBackButton";
import MoveNextButton from "../Components/MoveNextButton";

interface SpineExercise1ScreenProps {
  navigation: any;
}

const SpineExercise1Screen: React.FC<SpineExercise1ScreenProps> = () => {
  const navigation = useNavigation<StackNavigationProp<ParamListBase>>();
  const [round, setRound] = React.useState(0); // Current round
  const [timer, setTimer] = React.useState(30); // Initial timer value in seconds
  const [isRunning, setIsRunning] = React.useState(false); // Flag to indicate whether the timer is running

  // Function to move to the next round
  const nextRound = () => {
    setRound((prevRound) => Math.min(prevRound + 1, 5)); // Increment round, up to 5 rounds
    setTimer(30); // Reset timer to 30 seconds for the next round
    setIsRunning(true); // Start the timer automatically for the next round
  };

  // Function to start the timer
  const startTimer = () => {
    setIsRunning(true);
  };

  // Function to stop the timer
  const stopTimer = () => {
    setIsRunning(false);
  };

  React.useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isRunning) {
      interval = setInterval(() => {
        setTimer((prevTimer) => {
          if (prevTimer === 0) {
            stopTimer();
            nextRound(); // Move to the next round automatically after the timer ends
            return 30;
          }
          return prevTimer - 1;
        });
      }, 1000);
    } else {
      clearInterval(interval);
    }

    // Clear interval on unmount
    return () => clearInterval(interval);
  }, [isRunning, round]);

  // Function to calculate the arc path
 // Function to calculate the arc path
const getArcPath = (index: number) => {
  const smallerRadius = 40; // Radius of the smaller circle
  const startAngle = index * (360 / 5); // Start angle in degrees
  const endAngle = (index + 1) * (360 / 5); // End angle in degrees
  const startRad = startAngle * Math.PI / 180; // Convert start angle to radians
  const endRad = endAngle * Math.PI / 180; // Convert end angle to radians
  const x1 = Math.cos(startRad) * smallerRadius + 100; // X-coordinate of the starting point
  const y1 = Math.sin(startRad) * smallerRadius + 100; // Y-coordinate of the starting point
  const x2 = Math.cos(endRad) * smallerRadius + 100; // X-coordinate of the ending point
  const y2 = Math.sin(endRad) * smallerRadius + 100; // Y-coordinate of the ending point
  return `M ${x1},${y1} A ${smallerRadius},${smallerRadius} 0 ${(endAngle - startAngle) <= 180 ? "0" : "1"} 1 ${x2},${y2}`;
};


  return (
    <View style={styles.container}>
      <LinearGradient
        style={styles.neckExercise1Child}
        locations={[0.01, 1]}
        colors={["#add9d8", "rgba(217, 217, 217, 0)"]}
      />
      <View style={styles.topHeader}>
      <GoBackButton/>
      <Text style={styles.header} >Seated Side Bend </Text>
      <MoveNextButton targetScreen="SpineExercise2Screen"/>
      </View>
      <View style={styles.centerCard}>
      <Image
        style={styles.maskGroupIcon}
    
        source={require("./assets/images/seated-side-big.png")}
      />
      <View style={styles.timeRounds}>
      <View style={styles.timerContainer}> 
      <View style={{flexDirection:"row"}}>
        <Text style={styles.timerText}>Time Left: </Text>
        <Text style={styles.timerText}>{timer} seconds</Text>
        </View>
        <TouchableOpacity style={styles.button} onPress={isRunning ? stopTimer : startTimer} disabled={round === 5}>
        <Text style={styles.buttonText}>{isRunning ? "Stop" : "Start"}</Text>
      </TouchableOpacity>
      </View>
      <View style={styles.rounds}>
        <View style={styles.circle}>
        <Svg width="200" height="200">
          <Circle cx="100" cy="100" r="40" fill="none" stroke="#ADD9D6" strokeWidth="10" />
          {Array.from({ length: round }).map((_, index) => (
            <Path
              key={index}
              d={getArcPath(index)}
              fill="none"
              stroke="#3DD9D6"
              strokeWidth="10"
            />
          ))}
        </Svg>
        </View>
        <Text style={styles.roundText}>Round {round} / 5</Text>
        </View>
        </View>
        <Text style={styles.stepsTypo}>{`Steps:`}</Text>
        <View style={styles.textPart}>
        <Text style={styles.sitOrStand}>Sit on the floor with your legs extended straight in front of you.</Text> 
        <Text style={styles.sitOrStand}>Place your hands on the floor beside you for support.</Text>
        <Text style={styles.sitOrStand}>Inhale and lengthen your spine.</Text>
        <Text style={styles.sitOrStand}>Exhale as you gently lean to one side, bringing your ear toward your shoulder.</Text>
        <Text style={styles.sitOrStand}>Hold the stretch for 15-30 seconds, feeling the stretch along the side of your torso.</Text>
        <Text style={styles.sitOrStand}>Inhale and return to the center.</Text>
        <Text style={styles.sitOrStand}>Repeat on the other side.</Text>
        
      <Text style={styles.commentsHeader}>Comments:</Text>
      <Text style={styles.commentsText}>Improve flexibility in the spine and stretches the muscles along the sides of your torso. It can contribute to better posture and reduced stiffness in the lower back.</Text>
      </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Color.colorWhite,
    padding: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },

  topHeader:{
    flexDirection: 'row',
    alignContent: 'center',
    justifyContent:'space-between',
    top: 30,
  },
  header:{
    fontSize: 20,
    fontWeight: 'bold',
    paddingLeft: 30,
    paddingRight: 30,
  },
  centerCard:{
    width: 312,
    top: 50,
    height: 700,
    backgroundColor: '#ffffff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    overflow: 'hidden', // Ensure the shadow stays inside the card
    // Add the inner shadow
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 0,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  
  maskGroupIcon: {
    width: 280,
    height: 160,
    bottom: 50,
  },
  
  timeRounds:{
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    height: 150,
    bottom: 50,
  },
  timerContainer: {
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
    flexDirection: "column",
    left: 40,
  },
  roundText: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#000",
    bottom: 30,
  },
  rounds:{
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    left: 30,
  },
  circle:{
    height: 150,
    bottom: 30,
  },
  timerText: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#000",
  },
  neckExercise1Child: {
    width: 360,
    backgroundColor: "transparent",
    position: "absolute",
    height:700,
  },

  button: {
    backgroundColor: "#ADD9D6",
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    marginTop: 20,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#FFF",
  },
  stepsTypo: {
    fontSize: FontSize.size_mini,
    fontFamily: FontFamily.interMedium,
    fontWeight: "bold",
    bottom: 70,
    alignSelf: 'flex-start',
    left: 10,
  },
  sitOrStand: {
    fontSize: FontSize.eNHelperText_size,
    fontFamily: FontFamily.eNHelperText,
    textAlign: 'left',
    bottom: 70,
    paddingLeft: 10,
  },
  commentsText: {
    textAlign: "left",
    color: Color.colorBlack,
    bottom: 60,
    paddingHorizontal: 10,
    
  },
  textPart:{
    flexDirection:'column',

  },
  commentsHeader:{
    fontWeight: 'bold',
    textAlign: "left",
    color: Color.colorBlack,
    bottom: 65,
    left: 10,
  },

});

export default SpineExercise1Screen;
