import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import Button from '../Components/button';
import { LinearGradient } from 'expo-linear-gradient'; 
import { useNavigation, NavigationProp } from '@react-navigation/native';


interface FacetrackingScreenProps {
  navigation: any; 
}

const FacetrackingScreen: React.FC<FacetrackingScreenProps> = () => {
    const navigation = useNavigation<NavigationProp<any>>();
    const handleContinuePress = () => {
        navigation.navigate('DashboardScreen'); // Navigate to ProfilingScreen when signup button is pressed
      };
    return (
     <View style={styles.container} >
         <Image source={require('./assets/images/faceTrackingIllustration.png')} style={styles.profileImage} />
     <LinearGradient colors={['rgb(110, 154, 153)', 'rgb(110, 154, 153)']} style={styles.mainContainer} 
     start={{ x: 0.5, y: 0 }} 
     end={{ x: 0.5, y: 1 }} >
     <Text style={styles.description}>Keep your face in the center of the circle</Text>
     
      <TouchableOpacity  onPress={handleContinuePress}>
        <Button title="Continue"
              onPress={handleContinuePress}
              buttonStyle={styles.buttonStyle}
              textStyle={styles.buttonText}/>
      </TouchableOpacity>
      </LinearGradient>
    </View>
   
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  mainContainer:{
    flex: 1,
    height: '100%',
    width: '100%',
    borderTopLeftRadius: 70,
    borderTopRightRadius: 70,
    borderBottomLeftRadius: 0, // Set to 0 if you want only the top corners rounded
    borderBottomRightRadius: 0, // Set to 0 if you want only the top corners rounded
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  profileImage: {
    width: '100%',
    height: '20%',
    aspectRatio: 2,
  },
  
  description: {
    fontSize: 14,
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    padding: 20,
    paddingTop: 2,
    paddingBottom: 2,
  },
  

  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  buttonStyle:{
    backgroundColor: '#AFA58F',
    width: 143 ,
    height: 38,
    borderRadius: 15,
  },
 
});


export default FacetrackingScreen;
