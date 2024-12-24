import * as React from "react";
import { StyleSheet, Image, View, Text, Pressable, ImageBackground } from "react-native";
import { StackNavigationProp } from "@react-navigation/stack";
import { useNavigation, ParamListBase } from "@react-navigation/native";

import { Border, Color, FontSize, FontFamily } from "../GlobalStyles";
import GoBackButton from "../Components/goBackButton";
import BottomMenu from "../Components/bottomMenu";


interface PracticeScreenProps {
    navigation: any; 
  }
  
  const PracticeScreen: React.FC<PracticeScreenProps> = () => {
  const navigation = useNavigation<StackNavigationProp<ParamListBase>>();

  return (
    <View style={styles.container}>
       <View style={styles.header}>
      <GoBackButton/>
      <Text style={styles.practice}>Practice</Text>
      <Image
        style={styles.walkingIcon}
        source={require("./assets/images/walkingVector.png")}
      />
      </View>
      {/* top center card */}
<View style={styles.topContainer}>
      <Text
          style={styles.revitalizeYourPosture}
        >{`Revitalize your posture, boost confidence, and alleviate discomfort with our tailored exercises. 
Just minutes a day can transform how you look, feel, and live. Elevate your well-being effortlessly.
`}</Text>
</View>
      <Pressable
        // style={[styles.moveback, styles.movebackPosition]}
        onPress={() => navigation.navigate("SpineExerciseScreen")}
      >
      <View style={styles.card}>
          <Image
            style={styles.neckIcons}
            source={require("./assets/images/spinalVector.png")}
          />
          <Text style={styles.cardTitle}>Spine Exercise</Text>
          <Image
            style={styles.doubleLeftIcon}
            source={require("./assets/images/double-left.png")}
          />
          
        </View>
        </Pressable>
{/* second card */}
        <Pressable
        // style={[styles.moveback, styles.movebackPosition]}
        onPress={() => navigation.navigate("ShoulderExerciseScreen")}
      >
      <View style={styles.card}>
          <Image
            style={styles.neckIcons}
            source={require("./assets/images/backpainVector.png")}
          />
          <Text style={styles.cardTitle}>Shoulder Exercise</Text>
          <Image
            style={styles.doubleLeftIcon}
            source={require("./assets/images/double-left.png")}
          />
          
        </View>
        </Pressable>

        {/* thisrd card */}

        <Pressable
        // style={[styles.moveback, styles.movebackPosition]}
        onPress={() => navigation.navigate("NeckExerciseScreen")}
      >
      <View style={styles.card}>
          <Image
            style={styles.neckIcons}
            source={require("./assets/images/neckpainVector.png")}
          />
          <Text style={styles.cardTitle}>Neck Exercise</Text>
          <Image
            style={styles.doubleLeftIcon}
            source={require("./assets/images/double-left.png")}
          />
          
        </View>
        </Pressable>
       <ImageBackground
          style={styles.exercise21}     
          source={require("./assets/images/exerciseVector.png")}>
      
        </ImageBackground>
    <View style={styles.bottomMenu}>
    <BottomMenu/>
    </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignContent: 'center',
  },


  exercise21: {
    width: 313,
    height: 300,
    resizeMode: 'cover',
    left: 30,
    right: 0,
  },
  bottomMenu:{
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
  header:{
    flexDirection: 'row',
    top: 10,
  },
  practice: {
    left: 139,
    fontSize: FontSize.size_2xl,
    fontWeight: 'bold',
    width: 'auto',
    height: 27,
    textShadowRadius: 4,
    textShadowOffset: {
      width: 0,
      height: 4,
    },
    textShadowColor: "rgba(0, 0, 0, 0.25)",
    textAlign: "left",
    color: Color.colorBlack,
    fontFamily: FontFamily.eNHelperText,
    position: "absolute",
  },
  walkingIcon: {
    left: 110,
    height: 30,
    width: 30,
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
    width: 320,
    justifyContent: 'space-between',
    flexDirection: 'row',
    left: 20,
  },
  cardTitle: {
    fontSize: FontSize.size_lg,
    fontFamily: FontFamily.eNHelperText,
    color: Color.colorBlack,
    textAlign: "center",
    textAlignVertical: "center",
  },
  revitalizeYourPosture: {
    fontSize: FontSize.size_2xs,
    textAlign: "center", 
    justifyContent: 'space-between',
    paddingLeft: 10,
    paddingTop: 10,
    paddingRight: 10,
    color: Color.colorBlack,
    fontFamily: FontFamily.eNHelperText,
  },

  topContainer: {
    borderRadius: 25,
    backgroundColor: "rgba(255, 255, 255, 0.7)",
    marginTop: 20,
    marginBottom: 10,
    width: 290,
    left: 30,
    justifyContent: 'center',
    shadowOpacity: 1,
    elevation: 4,
    shadowRadius: 4,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowColor: "rgba(0, 0, 0, 0.25)",
  },
});

export default PracticeScreen;
