import * as React from "react";
import { Text, Image , StyleSheet, View, Pressable, Dimensions } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";
import { useNavigation,NavigationProp, ParamListBase } from "@react-navigation/native";
import BottomMenu from '../Components/bottomMenu';
import { FontAwesome } from '@expo/vector-icons';
import { BarChart } from 'react-native-chart-kit';

import { FontFamily, Color, Border, FontSize } from "../GlobalStyles";

interface AnalysisScreenProps {
    navigation: any; 
  }

const AnalysisScreen: React.FC<AnalysisScreenProps>  = () => {
    const navigation = useNavigation<NavigationProp<any>>();
    const [totalActiveTime, setTotalActiveTime] = React.useState<number>(0);
    const [goodPostureTime, setGoodPostureTime] = React.useState<number>(0);

    // Simulated function to calculate total active time
    const calculateTotalActiveTime = () => {
      // Replace this with your actual logic to calculate active time
      // For demonstration, let's simulate a random value between 0 and 100
      const randomActiveTime = Math.floor(Math.random() * 100);
      setTotalActiveTime(randomActiveTime);
    };

    // Simulated function to calculate good posture time
    const calculateGoodPostureTime = () => {
      // Replace this with your actual logic to calculate good posture time
      // For demonstration, let's simulate a random value between 0 and 100
      const randomGoodPostureTime = Math.floor(Math.random() * 100);
      setGoodPostureTime(randomGoodPostureTime);
    };
  
    // Use useEffect to update active time and good posture time when component mounts
    React.useEffect(() => {
      calculateTotalActiveTime();
      calculateGoodPostureTime();
    }, []);

    const [totalAlerts, setTotalAlerts] = React.useState<number>(0);

  // Function to simulate counting total alerts
  const countTotalAlerts = () => {
    // Replace this with your actual logic to count alerts
    // For demonstration, let's simulate a random value between 0 and 10
    const randomAlerts = Math.floor(Math.random() * 10);
    setTotalAlerts(randomAlerts);
  };

  // Use useEffect to update total alerts when component mounts
  React.useEffect(() => {
    countTotalAlerts();
  }, []);

  const [thisWeekScore, setThisWeekScore] = React.useState<number>(0);
  const [lastWeekScore, setLastWeekScore] = React.useState<number>(0);

  // Function to simulate calculating posture scores for this and last week
  const calculatePostureScores = () => {
    // Replace this with your actual logic to calculate scores
    // For demonstration, let's simulate random values between 0 and 100
    const randomThisWeekScore = Math.floor(Math.random() * 100);
    const randomLastWeekScore = Math.floor(Math.random() * 100);
    setThisWeekScore(randomThisWeekScore);
    setLastWeekScore(randomLastWeekScore);
  };

  // Use useEffect to update scores when component mounts
  React.useEffect(() => {
    calculatePostureScores();
  }, []);

  const generateDataForGraph = () => {
    // Sample data for demonstration
    return [
      { day: 'Mon', thisWeekScore: 80, lastWeekScore: 70 },
      { day: 'Tue', thisWeekScore: 75, lastWeekScore: 72 },
      { day: 'Wed', thisWeekScore: 90, lastWeekScore: 85 },
      { day: 'Thu', thisWeekScore: 85, lastWeekScore: 80 },
      { day: 'Fri', thisWeekScore: 88, lastWeekScore: 82 },
      { day: 'Sat', thisWeekScore: 92, lastWeekScore: 78 },
      { day: 'Sun', thisWeekScore: 95, lastWeekScore: 90 },
    ];
  };


  return (
    <View style={styles.container}>
     
        <Text style={styles.header}>Weekly Overview</Text>
      
    <View style={styles.card}>
     <View style={styles.firstTab}>
        <View style={styles.timeTab}>
          <FontAwesome name="clock-o" size={30} color="#026761" style={styles.icon} />
          <Text style={styles.activeTime}>{totalActiveTime} minutes</Text>
          <Text style={styles.title}>Total Active Time</Text>
        </View>
 {/* Separator */}
        <View style={styles.separator} />


        <View style={styles.timeTab}>
          <Image style={styles.icon}
          source={require('./assets/images/postureVector.png')} />
          <Text style={styles.activeTime}>{goodPostureTime} minutes</Text>
          <Text style={styles.title}>Good Posture Time</Text>
        </View>
{/* Separator */}
        <View style={styles.separator} />

        
        <View style={styles.timeTab}>
          <FontAwesome name="bell" size={30} color="#026761" style={styles.icon}/>
          <Text style={styles.activeTime}>{totalAlerts} </Text>
          <Text style={styles.title}>Total Alerts</Text>
        </View>
     </View>


      <View style={styles.secondTab}>
        <View style={styles.scoreCard}>
          <FontAwesome name="trophy" size={35} color="#026761" style={styles.icon}/>
          <Text style={styles.score}>{thisWeekScore}</Text>
          <Text style={styles.scoreTitle}>Posture Score</Text>
        </View>
        <View style={styles.comparison}>
        <Text style={styles.comparisonText}>Comparison with Last Week: {thisWeekScore > lastWeekScore ? "Improved" : "Decreased"}</Text>
        <Text style={styles.comparisonText}>Last Week's Score: {lastWeekScore}</Text>
      </View>
      </View>
   
      <BarChart
    data={{
      labels: generateDataForGraph().map(item => item.day),
      datasets: [
        {
          data: generateDataForGraph().map(item => item.thisWeekScore),
          color: (opacity = 1) => `rgba(0, 0, 255, ${opacity})`, // Light blue color for this week's score
        },
        {
          data: generateDataForGraph().map(item => item.lastWeekScore),
          color: (opacity = 1) => `rgba(255, 0, 0, ${opacity})`, // Blood red color for last week's score
        },
      ],
    }}
    width={300} // Width of the chart
    height={200} // Height of the chart
    yAxisLabel="" // Label for y-axis
    yAxisSuffix="" // No suffix for y-axis values
    chartConfig={{
      backgroundColor: '#FFFFFF', // Background color of the chart
      backgroundGradientFrom: '#FFFFFF', // Background gradient start color
      backgroundGradientTo: '#FFFFFF', // Background gradient end color
      decimalPlaces: 0, // Number of decimal places
      color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`, // Color of labels
      labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`, // Color of values
      style: {
        borderRadius: 8,
        
      },
    }}
    style={{
      marginVertical: 4,
      borderRadius: 8,
      // right: 30,
      alignSelf: 'center',
      
    }}
  />


    </View>
    <View style={styles.bottomMenu}>
    <BottomMenu/>
    </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container:{
    flex: 1,
    backgroundColor: '#EAF0F2', 
  },
  header:{
    alignSelf:'center',
    top: 10,
    fontSize: 18,
    fontWeight: 'bold',
  },
  bottomMenu:{
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
  card: {
    width: 340,
    flexDirection: 'column',
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    paddingRight: 30,
    paddingLeft: 30,
    paddingBottom: 10,
    paddingTop: 10,
    marginTop: 70,
    elevation: 3, // Shadow for Android
    shadowColor: '#000000', // Shadow for iOS
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    justifyContent: 'center',
    alignSelf: 'center',
  },
  separator: {
    width: 1,
    backgroundColor: 'black',
    marginHorizontal: 2,
    marginVertical: 2,

  },
  timeTab:{
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 5,
    paddingLeft: 10,
    paddingRight: 10,
  },
  scoreCard:{
    backgroundColor: 'rgba(175, 165, 143, 0.89)',
    borderRadius: 20,
    padding: 20,
    width: 150,
    alignItems: 'center',
    marginTop: 10,
    elevation: 3, // Shadow for Android
    shadowColor: '#000000', // Shadow for iOS
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  firstTab:{
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondTab:{
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 9,
    fontWeight: 'bold',
  },
  activeTime: {
    fontSize: 11,
  },
  icon: {
    justifyContent: 'center',
    alignContent: 'center',
  },
  score: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#F56D0A', // Color for posture score
    marginTop: 5,
  },
  scoreTitle:{
    fontSize: 12,
    fontWeight: 'bold',
  },
  
  comparison: {
    alignItems: 'center',
  },
  comparisonText: {
    fontSize: 12,
    marginTop: 10,
  },

});

export default AnalysisScreen;
