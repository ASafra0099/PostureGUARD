import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import Button from '../Components/button';
import { LinearGradient } from 'expo-linear-gradient'; 
import { useNavigation, NavigationProp } from '@react-navigation/native';
import GoBackButton from '../Components/goBackButton';


interface BindingScreen1Props {
  navigation: any; 
}

const BindingScreen1: React.FC<BindingScreen1Props> = () => {
  const navigation = useNavigation<NavigationProp<any>>();
 
    const handleContinuePress = () => {
        navigation.navigate('BindingScreen2'); // Navigate to ProfilingScreen when signup button is pressed
      };
    return (
     <View style={styles.container} >
     <LinearGradient colors={['rgb(178,169,148)', 'rgb(178,169,148)']} style={styles.container} 
     start={{ x: 0.5, y: 0 }} 
     end={{ x: 0.5, y: 1 }} >
      <View style={styles.header}>
        <GoBackButton/>
      </View>
      <Text style={styles.title}>Bind Camera</Text>
      <Text style={styles.description1}>If you have a camera, you can pair it with your phone here.</Text>
      <View>
      <Text style={styles.description2}>Please make sure Bluetooth is turned on in both camera and phone.</Text>
      <Text style={styles.description2}>Put your phone as close to the device as possible.</Text>
      <Text style={styles.description2}>After that, click the button below to bind.</Text>
      <Text style={styles.description2}>Select your camera from the list of available devices to initiate pairing.</Text></View>
      <Image source={require('./assets/images/bindCamImages.png')} style={styles.bindingImage} />
   
      <TouchableOpacity  onPress={handleContinuePress}>
        <Button title="Click to bind"
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
    width: '100%',
  },
  title: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,

  },
  header:{
    alignSelf: 'flex-start',
    marginBottom: 15,

  },
  bindingImage: {
    width: '100%',
    height: 200,
    marginTop: 60,
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
    fontWeight: 'bold',
  },
  description2: {
    fontSize: 14,
    color: 'white',
    fontWeight: 'bold',
    paddingLeft: 10,
    paddingRight: 10,
    paddingBottom: 2,
  },

  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  buttonStyle:{
    width: 183,
    height: 50,
    borderRadius: 29,
    backgroundColor: '#3D9D9B',
    // Shadow properties
    shadowColor: '#346D6B',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.5,
    shadowRadius: 4,
    elevation: 5, // For Android
    // End of shadow properties
    justifyContent: 'center',
    alignItems: 'center',
  },

});

export default BindingScreen1;
