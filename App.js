import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import 'react-native-gesture-handler';




import SplashScreen from './screens/SplashScreen';
import PopupScreen from './screens/PopupScreen';
import SignupScreen from './screens/SignupScreen';
import LoginScreen from './screens/LoginScreen';
// import ProfileScreen from './screens/ProfilingScreen';
// import FacetrackingScreen from './screens/FacetrackingScreen';
import DashboardScreen from './screens/DashboardScreen';
import BindingScreen1 from './screens/BindingScreen1';
import BindingScreen2 from './screens/BindingScreen2';
import AnalysisScreen from './screens/AnalysisScreen';
import SettingsScreen from './screens/SettingsScreen';
import SupportHelpScreen from './screens/SupportHelpScreen';
import AccountSettingsScreen from './screens/AccountSettingsScreen';
import PracticeScreen from './screens/PracticeScreen';
import NeckExerciseScreen from './screens/NeckExerciseScreen';
import NeckExercise1Screen from './screens/NeckExercise1Screen';
import NeckExercise2Screen from './screens/NeckExercise2Screen';
import NeckExercise3Screen from './screens/NeckExercise3Screen';
import SpineExerciseScreen from './screens/SpineExerciseScreen';
import SpineExercise1Screen from './screens/SpineExercise1Screen';
import SpineExercise2Screen from './screens/SpineExercise2Screen';
import SpineExercise3Screen from './screens/SpineExercise3Screen';
import ShoulderExerciseScreen from './screens/ShoulderExerciseScreen';
import ShoulderExercise1Screen from './screens/ShoulderExercise1Screen';
import ShoulderExercise2Screen from './screens/ShoulderExercise2Screen';
import ShoulderExercise3Screen from './screens/ShoulderExercise3Screen';

const Stack = createStackNavigator();

const App = () => {



  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="SplashScreen" component={SplashScreen} />
        <Stack.Screen name="PopupScreen" component={PopupScreen} />
        <Stack.Screen name="SignupScreen" component={SignupScreen} />
        <Stack.Screen name="LoginScreen" component={LoginScreen} />
        {/* <Stack.Screen name="ProfilingScreen" component={ProfileScreen} /> */}
        {/* <Stack.Screen name="FacetrackingScreen" component={FacetrackingScreen} /> */}
        <Stack.Screen name="DashboardScreen" component={DashboardScreen} />
        <Stack.Screen name="BindingScreen1" component={BindingScreen1} />
        <Stack.Screen name="BindingScreen2" component={BindingScreen2} />
        <Stack.Screen name="AnalysisScreen" component={AnalysisScreen} />
        <Stack.Screen name="SettingsScreen" component={SettingsScreen} />
        <Stack.Screen name="SupportHelpScreen" component={SupportHelpScreen}/>
        <Stack.Screen name="AccountSettingsScreen" component={AccountSettingsScreen}/>
        <Stack.Screen name="PracticeScreen" component={PracticeScreen}/>
        <Stack.Screen name="NeckExerciseScreen" component={NeckExerciseScreen}/>
        <Stack.Screen name="NeckExercise1Screen" component={NeckExercise1Screen}/>
        <Stack.Screen name="NeckExercise2Screen" component={NeckExercise2Screen}/>
        <Stack.Screen name="NeckExercise3Screen" component={NeckExercise3Screen}/>
        <Stack.Screen name="SpineExerciseScreen" component={SpineExerciseScreen}/>
        <Stack.Screen name="SpineExercise1Screen" component={SpineExercise1Screen}/>
        <Stack.Screen name="SpineExercise2Screen" component={SpineExercise2Screen}/>
        <Stack.Screen name="SpineExercise3Screen" component={SpineExercise3Screen}/>
        <Stack.Screen name="ShoulderExerciseScreen" component={ShoulderExerciseScreen}/>
        <Stack.Screen name="ShoulderExercise1Screen" component={ShoulderExercise1Screen}/>
        <Stack.Screen name="ShoulderExercise2Screen" component={ShoulderExercise2Screen}/>
        <Stack.Screen name="ShoulderExercise3Screen" component={ShoulderExercise3Screen}/>


        
      </Stack.Navigator>
    </NavigationContainer>
  );
};


export default App;
