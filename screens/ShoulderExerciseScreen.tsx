import React from "react";
import { StyleSheet, View, Text, Image, Pressable } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";
import { useNavigation, ParamListBase } from "@react-navigation/native";
import { Border, Color, FontFamily, FontSize } from "../GlobalStyles";
import GoBackButton from "../Components/goBackButton";

interface ShoulderExerciseScreenProps {
  navigation: StackNavigationProp<ParamListBase>;
}

const ShoulderExerciseScreen: React.FC<ShoulderExerciseScreenProps> = ({ navigation }) => {
    
  return (
    <View style={styles.container}>
      {/* Header Card */}
      <LinearGradient
          colors={["#ADD9D6", "#D9D9D9"]}
          style={styles.topCard}
          start={{ x: 0, y: 0.5 }}
          end={{ x: 0, y: 1 }}
        >
      
       <View style={styles.topHeader}>
        <GoBackButton />
        <Text style={styles.header}>Shoulder Exercises</Text>
        </View>
        <Image
          style={styles.neck1Icon}
          source={require("./assets/images/shoulderVector.png")}
        />
        {/* Add other icons and text here */}
        <View style={styles.topIcons}>
        <View style={styles.eclipsesandIcons}>
        <Image
          style={styles.ellipseIcon}
          source={require("./assets/images/ellipse-239.png")}
        />
        <Image
          style={styles.barbellIcon}
          source={require("./assets/images/barbell.png")}
        />
      </View>

      {/* Second ellipse and icon */}
      <View style={styles.eclipsesandIcons}>
        <Image
          style={styles.ellipseIcon}
          source={require("./assets/images/ellipse-239.png")}
        />
        <Image
          style={styles.deliveryTimeIcon}
          source={require("./assets/images/delivery-time.png")}
        />
      </View>
      </View>

      {/* total workout */}
      <View style={styles.workOutContainer}>
      <Text >Total Workout</Text>
      <Text>Overall time</Text>
      </View>
      <View style={styles.workOutContainer}>
      <Text style={[styles.timeBox, styles.timerText]}>0</Text>
      <Text style={[styles.timeBox, styles.timerText]}>0 s</Text>
      </View>
      

      </LinearGradient>

      {/* Card for "Up and Down Nods" */}
      <Pressable
        // style={[styles.moveback, styles.movebackPosition]}
        onPress={() => navigation.navigate("ShoulderExercise1Screen")}
      >
      <LinearGradient
          colors={["#ADD9D8", "#3DD9D6"]}
          style={styles.card}
          start={{ x: 1, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <Image
            style={styles.neckIcons}
            source={require("./assets/images/sh rotation 1.png")}
          />
          <Text style={styles.cardTitle}>Shoulder Rotation</Text>
          <Image
            style={styles.doubleLeftIcon}
            source={require("./assets/images/double-left.png")}
          />
          
        </LinearGradient>
        </Pressable>

      {/* Card for "Side To Side Turns" */}
      <Pressable
        // style={[styles.moveback, styles.movebackPosition]}
        onPress={() => navigation.navigate("ShoulderExercise2Screen")}
      >
       <LinearGradient
          colors={["#ADD9D8", "#3DD9D6"]}
          style={styles.card}
          start={{ x: 1, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
      <Image
        style={styles.neckIcons}
        source={require("./assets/images/Sh rolls 1.png")}
      />
        <Text style={styles.cardTitle}>Shoulder Rolls </Text>
        <Image
        style={styles.doubleLeftIcon}
        source={require("./assets/images/double-left.png")}
      />
      </LinearGradient>
      </Pressable>

      {/* Card for "Side To Side Tilts" */}
      <Pressable
        // style={[styles.moveback, styles.movebackPosition]}
        onPress={() => navigation.navigate("ShoulderExercise3Screen")}
      >
       <LinearGradient
          colors={["#ADD9D8", "#3DD9D6"]}
          style={styles.card}
          start={{ x: 1, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
      <Image
        style={styles.neckIcons}
        source={require("./assets/images/torso twist 1.png")}
      />
        <Text style={styles.cardTitle}>Torso Twist</Text>
        <Image
        style={styles.doubleLeftIcon}
        source={require("./assets/images/double-left.png")}
      />
      </LinearGradient>
      </Pressable>
      <Image
        style={styles.frameIcon}
        source={require("./assets/images/frame-22.png")}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Color.colorWhite,
    padding: 10,
  },
  topCard:{
    backgroundColor: Color.colorWhite,
    borderRadius: Border.br_20xl,
    elevation: 3,
    shadowColor: Color.colorBlack,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    marginBottom: 20,
    padding: 20,
    height: 'auto',
  },
  card: {
    backgroundColor: Color.colorWhite,
    borderRadius: Border.br_20xl,
    elevation: 3,
    shadowColor: Color.colorBlack,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    marginBottom: 15,
    padding: 15,
    height: 60,
    justifyContent: 'space-between',
    flexDirection: 'row',
    
  },
  topHeader:{
    flexDirection: 'row',

  },
  header: {
    fontSize: FontSize.size_xl,
    fontFamily: FontFamily.eNHelperText,
    color: Color.colorBlack,
    textAlign: "center",
    fontWeight: 'bold',
    paddingLeft: 35,
  },
  icon: {
    width: 30,
    height: 30,
  },
  neck1Icon: {
    width: 100,
    height: 100,
    alignSelf: "center",
    
  },
  topIcons:{
    flexDirection:'row',
    justifyContent: 'space-around',
    alignItems: 'center',

  },
  eclipsesandIcons: {
    position: 'relative',
    alignItems: 'center',

  },
  ellipseIcon: {
    width: 50, 
    height: 50, 

  },
  barbellIcon: {
    position: 'absolute',
    top: 10,
    width: 25, 
    height: 25, 
    
  },
  deliveryTimeIcon: {
    position: 'absolute',
    top: 10, 
    width: 25, 
    height: 25, 
    
  },
  
  workOutContainer:{
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  timerText:{
    fontWeight: 'bold',
    textAlign: 'center',
    textAlignVertical: 'center',
    color: '#3498db',
  },
  timeBox:{
    height: 21,
    width: 60,
    textAlign: "center",
    fontSize: 14,
    color:"#333",
    borderRadius: 10,
    marginTop: 10,
    borderColor:'#ddd',
    shadowColor: "#000",
    elevation: 2,
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowRadius: 2.84,
    shadowOpacity: 0.25,
    borderTopWidth: 1,
    borderLeftWidth: 1,
    borderRightWidth: 1,
},  
  cardTitle: {
    fontSize: FontSize.size_lg,
    fontFamily: FontFamily.eNHelperText,
    color: Color.colorBlack,
    textAlign: "center",
    textAlignVertical: "center",
  },
  frameIcon: {
    top: 490,
    left: -119,
    width: 541,
    height: 380,
    position: "absolute",

  },
  neckIcons:{
    width: 50,
    height: 44,
    backgroundColor: 'transparent',
    alignSelf: "center",
  },
  doubleLeftIcon:{
    width: 20,
    height: 20,
    backgroundColor: 'transparent',
    alignSelf: "center",
  },
});

export default ShoulderExerciseScreen;
