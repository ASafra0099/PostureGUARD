import * as React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { createMaterialTopTabNavigator, MaterialTopTabBarProps } from '@react-navigation/material-top-tabs';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, NavigationProp } from '@react-navigation/native';

const Tab = createMaterialTopTabNavigator();

const AllFeaturesScreen = () => (
  <View style={styles.screen}>
    <Text style={styles.title}>Welcome to Posture Guard!</Text>
    <Image source={require("./assets/images/popup1.png")}
    style={styles.illustartion}/>
    
  </View>
);

const FaceIdentificationScreen = () => (
  <View style={styles.screen}>
    <Text style={styles.title2}>FaceFit Identification</Text>
    <Image source={require("./assets/images/popup2.png")}
    style={styles.illustartion}/>
  </View>
);

const CameraConnectionScreen = () => (
  <View style={styles.screen}>
    <Text style={styles.title3}>Seamless Connection, Superior Posture</Text>
    <Image source={require("./assets/images/popup3.png")}
    style={styles.illustartion}/>
  </View>
);


const PopupScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp<any>>();
  const handleCloseBtn = () => {
    navigation.navigate('SignupScreen');
  };
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={[styles.getStart, styles.getBox]}>Let's Get Started</Text>
      <TouchableOpacity onPress={handleCloseBtn} style={styles.closeButton}>
        <Ionicons name="close" size={24} color="black" />
      </TouchableOpacity>
      </View>
      <Tab.Navigator
        tabBarPosition="bottom"
        tabBar={(props) => <CustomTabBar {...props} />}
      >
        <Tab.Screen name="All Features" component={AllFeaturesScreen} />
        <Tab.Screen name="Face Identification" component={FaceIdentificationScreen} />
        <Tab.Screen name="Camera Connection" component={CameraConnectionScreen} />
      </Tab.Navigator>
    </View>
  );
};

const CustomTabBar: React.FC<MaterialTopTabBarProps> = ({ state, descriptors, navigation }) => {
  return (
    <View style={styles.tabBar}>
      {state.routes.map((route, index) => {
        const isFocused = state.index === index;

        return (
          <TouchableOpacity
            key={route.key}
            onPress={() => navigation.navigate(route.name)}
            style={{ flex: 1 }}
          >
            <View style={[ { borderBottomWidth: isFocused ? 10 : 0, borderBottomColor: '#125488',borderRadius: 10}]}>
             
            </View>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1 ,
    backgroundColor: "rgba(173, 217, 216, 0.4)",
    alignContent: "center",
  },
  header:{
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 5,
    paddingRight: 15,
    paddingLeft: 15,
  },
  getStart:{
    fontSize: 12,
    marginTop: 10,
  },
  getBox:{
    width: 140,
    backgroundColor: '#125488',
    borderRadius: 10,
    display: 'flex',
    textAlign: 'center',
    textAlignVertical: 'center',
    fontWeight: 'bold',
    color:'#ffffff',
  },
  closeButton: {
    height: 40,
    top: 5,
    },
  screen: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: "rgba(173, 217, 216, 0.4)",
   paddingBottom: 30,
  },
  title:{
    fontSize: 24,
    fontWeight: "900" ,
    alignSelf: 'flex-start',
    paddingLeft: 40,
    marginRight: 100,
    textShadowColor: 'rgba(0, 0, 0, 0.3)', // Shadow color
    textShadowOffset: { width: 2, height: 2 }, // Shadow offset
    textShadowRadius: 4, 
  },
  title3:{
    fontSize: 24,
    fontWeight: "900" ,
    alignSelf: 'flex-start',
    paddingLeft: 40,
    marginRight: 10,
    textShadowColor: 'rgba(0, 0, 0, 0.3)', // Shadow color
    textShadowOffset: { width: 2, height: 2 }, // Shadow offset
    textShadowRadius: 4, 
    paddingBottom: 5, 
  },  
  title2:{
    fontSize: 24,
    fontWeight: "900" ,
    alignSelf: 'flex-start',
    paddingLeft: 40,
    marginRight: 10,
    textShadowColor: 'rgba(0, 0, 0, 0.3)', // Shadow color
    textShadowOffset: { width: 2, height: 2 }, // Shadow offset
    textShadowRadius: 4, 
  },  

  illustartion:{
    height: 507,
    width: 294,
    resizeMode:'contain',
  },
 
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#D9D9D9',
    elevation: 0, // Remove shadow on Android
    shadowOpacity: 0, // Remove shadow on iOS
    borderTopWidth: 1,
    borderTopColor: '#cccccc',
    width: 250,
    alignSelf: 'center',
    borderRadius: 10,
    bottom: 30,
  },

});

export default PopupScreen;
