import * as React from "react";
import { StyleSheet, View, Image, Pressable, Text } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";
import { useNavigation, ParamListBase } from "@react-navigation/native";
import { Border, Color, FontFamily, FontSize } from "../GlobalStyles";

const PracticeSection: React.FC = () => {
  const navigation = useNavigation<StackNavigationProp<ParamListBase>>();

  return (
    <View style={styles.rectangleParent}>
      <View style={[styles.frameChild, styles.frameLayout]} />
      <LinearGradient
        style={[styles.frameItem, styles.frameLayout]}
        locations={[0, 1]}
        colors={["rgba(106, 158, 153, 0.5)", "rgba(106, 158, 153, 0)"]}
      />
      <Image
        style={styles.frameInner}
       
        source={require("../assets/activescreen.png")}
      />
      <Pressable
        style={[styles.vector, styles.vectorPosition]}
        onPress={() => navigation.navigate("Dashboard")}
      >
        <Image
          style={[styles.icon, styles.iconLayout]}
        
          source={require("../assets/vector2.png")}
        />
      </Pressable>
      <Pressable
        style={[styles.vector1, styles.vectorPosition]}
        onPress={() => navigation.navigate("Reccomendation")}
      >
        <Image
          style={[styles.icon, styles.iconLayout]}
  
          source={require("../assets/vector4.png")}
        />
      </Pressable>
      <Pressable
        style={[styles.settingLine, styles.vectorPosition]}
        onPress={() => navigation.navigate("Settings")}
      >
        <Image
          style={[styles.icon, styles.iconLayout]}
     
          source={require("../assets/setting-line1.png")}
        />
      </Pressable>
      <Pressable
        style={styles.chartAlt}
        onPress={() => navigation.navigate("Analytics")}
      >
        <Image
          style={[styles.chartAltChild, styles.iconLayout]}
          
          source={require("../assets/vector-8.png")}
        />
        <View style={styles.chartAltItem} />
      </Pressable>
      <Text style={[styles.analysis, styles.analysisTypo]}>{`Analysis
`}</Text>
      <Text style={[styles.settings, styles.analysisTypo]}>Settings</Text>
      <Text style={[styles.realTime, styles.analysisTypo]}>Real time</Text>
      <Text style={[styles.practice, styles.analysisTypo]}>Practice</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  frameLayout: {
    borderRadius: Border.br_21xl,
    left: 0,
    top: 0,
    height: 109,
    width: 388,
    position: "absolute",
  },
  vectorPosition: {
    top: "7.34%",
    position: "absolute",
  },
  iconLayout: {
    maxHeight: "100%",
    overflow: "hidden",
    maxWidth: "100%",
  },
  analysisTypo: {
    height: 21,
    width: 59,
    textAlign: "left",
    color: Color.colorWhite,
    fontFamily: FontFamily.eNHelperText,
    fontSize: FontSize.eNHelperText_size,
    top: 53,
    position: "absolute",
  },
  frameChild: {
    backgroundColor: Color.colorSteelblue_200,
  },
  frameItem: {
    backgroundColor: "transparent",
  },
  frameInner: {
    top: 3,
    left: 222,
    width: 45,
    height: 45,
    position: "absolute",
  },
  icon: {
    height: "100%",
    width: "100%",
  },
  vector: {
    left: "11.08%",
    right: "81.19%",
    bottom: "66.24%",
    width: "7.73%",
    height: "26.42%",
  },
  vector1: {
    left: "60.31%",
    right: "33.76%",
    bottom: "64.13%",
    width: "5.93%",
    height: "28.53%",
  },
  settingLine: {
    left: "81.96%",
    right: "10.98%",
    bottom: "63.76%",
    width: "7.06%",
    height: "28.9%",
  },
  chartAltChild: {
    height: "33.43%",
    width: "41.67%",
    top: "37.43%",
    right: "29.17%",
    bottom: "29.14%",
    left: "29.17%",
    position: "absolute",
  },
  chartAltItem: {
    height: "80.86%",
    width: "80.56%",
    top: "9.71%",
    right: "9.72%",
    bottom: "9.43%",
    left: "9.72%",
    borderRadius: Border.br_11xs,
    borderStyle: "solid",
    borderColor: Color.colorWhite,
    borderWidth: 2,
    position: "absolute",
  },
  chartAlt: {
    top: 8,
    left: 130,
    width: 36,
    height: 35,
    position: "absolute",
  },
  analysis: {
    left: 126,
  },
  settings: {
    left: 310,
  },
  realTime: {
    left: 27,
  },
  practice: {
    left: 223,
  },
  rectangleParent: {
    top: 723,
    left: -14,
    height: 109,
    width: 388,
    position: "absolute",
  },
});

export default PracticeSection;
