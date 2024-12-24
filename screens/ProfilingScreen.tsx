import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import Button from '../Components/button';
import { LinearGradient } from 'expo-linear-gradient'; 
import { useNavigation, NavigationProp } from '@react-navigation/native';


interface ProfileScreenProps {
  navigation: any; 
}

const ProfileScreen: React.FC<ProfileScreenProps> = () => {
  const navigation = useNavigation<NavigationProp<any>>();
 
    const handleContinuePress = () => {
        navigation.navigate('FacetrackingScreen'); // Navigate to ProfilingScreen when signup button is pressed
      };
    return (
     <View style={styles.container} >
     <LinearGradient colors={['rgb(110, 154, 153)', 'rgb(110, 154, 153)']} style={styles.container} 
     start={{ x: 0.5, y: 0 }} 
     end={{ x: 0.5, y: 1 }} >
      <Text style={styles.title}>Enrol Facial Info</Text>
      <Text style={styles.description1}>For the best results, keep the device 20cm to 50cm from you face and avoid the surroundings that are too bright or too dim when enrolling.</Text>
      <Text style={styles.description2}>This info will be used for user profiling, this is an important step before getting further into the application.</Text>
      <Image source={require('./assets/images/profilingIllustration.png')} style={styles.profileImage} />
     <View style={styles.sheildAnedtext} >
      <Image source={require('./assets/images/sheildIcon.png')} style={styles.sheildIcon} /> 
      <Text style={styles.description3}>Facial info is stored in the security module of this application and cannot be obtained by other apps</Text>
      </View>
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
    // backgroundColor: '#F5F5F5',
    // padding: 20,
    // paddingTop: 2,
    // paddingBottom: 2,
  },
  title: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  profileImage: {
    width: 200,
    height: 200,
    marginBottom: 20,
   
  },
  description1: {
    fontSize: 16,
    color: 'white',
    textAlign: 'center',
    marginBottom: 20,
    padding: 20,
    paddingTop: 2,
    paddingBottom: 2,
  },
  description2: {
    fontSize: 14,
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    padding: 20,
    paddingTop: 2,
    paddingBottom: 2,
  },
  description3: {
    fontSize: 12,
    color: 'white',
    textAlign: 'center',
    marginBottom: 20,
    // padding: 20,
    paddingTop: 2,
    paddingRight: 20,
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
  sheildIcon:{
   
   marginLeft: 15,
  },

  sheildAnedtext:{
    flexDirection: 'row',
  },
});

export default ProfileScreen;
