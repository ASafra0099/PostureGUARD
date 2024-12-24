import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity, ScrollView, Image, KeyboardAvoidingView, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation, NavigationProp } from '@react-navigation/native';
import * as ImagePicker from 'react-native-image-picker';

import Button from '../Components/button'; // Import the Button component

const SignupScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp<any>>();
  const [imageUri, setImageUri] = useState<string | null>(null);

  const handleLoginLinkPress = () => {
    navigation.navigate('LoginScreen');
  };

  const handleSignupPress = () => {
    navigation.navigate('DashboardScreen');
    // Handle signup logic here
  };

  const handleChoosePhoto = () => {
    // Handle choose photo logic
  };
  
  const handleTakePhoto = () => {
    // Handle take photo logic
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior="padding">
      <LinearGradient
       colors={['rgba(175, 165, 143, 0.55)', 'rgba(175, 165, 143, 0.50)']}
       style={styles.linearGradient}
       start={{ x: 0.5, y: 0 }}
       end={{ x: 0.5, y: 1 }}
      >
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          <Image source={require('./assets/images/registrationTopEclipse.png')} style={styles.topImage} />
          <View style={styles.registrationCenterBox}>
            <Text style={styles.signupTitle}>Signup</Text>
            <ScrollView contentContainerStyle={styles.scrollViewContent}>
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Name</Text>
                <TextInput placeholder="Name" style={styles.input} />
              </View>
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Age</Text>
                <TextInput placeholder="Age" style={styles.input} keyboardType="numeric" />
              </View>
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Sex</Text>
                <TextInput placeholder="Sex" style={styles.input} />
              </View>
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Email</Text>
                <TextInput placeholder="Email" style={styles.input} keyboardType="email-address" />
              </View>
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Password</Text>
                <TextInput placeholder="Password" style={styles.input} secureTextEntry />
              </View>
              <View style={styles.imagePickerContainer}>
                <Text style={styles.label}>Profile Picture</Text>
                {imageUri && (
                  <Image source={{ uri: imageUri }} style={styles.profileImage} />
                )}
                <View style={styles.buttonContainer}>
                  <TouchableOpacity onPress={handleChoosePhoto} style={styles.photoButton}>
                    <Text style={styles.photoButtonText}>Choose Photo</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={handleTakePhoto} style={styles.photoButton}>
                    <Text style={styles.photoButtonText}>Take Photo</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </ScrollView>
          </View>
          <Button
            title="Sign Up"
            onPress={handleSignupPress}
            buttonStyle={styles.signupButton}
            textStyle={styles.signupButtonText}
          />
          <View style={styles.loginLinkContainer}>
            <Text style={styles.alreadyText}>Already have an account?</Text>
            <TouchableOpacity onPress={handleLoginLinkPress}>
              <Text style={styles.loginText}>LogIn</Text>
            </TouchableOpacity>
          </View>
          <Image source={require('./assets/images/registrationIllustration.png')} style={styles.bottomImage} />
        </ScrollView>
      </LinearGradient>
    </KeyboardAvoidingView>
  );
};

const windowHeight = Dimensions.get('window').height;
const windowWidth = Dimensions.get('window').width;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  linearGradient: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  scrollContainer: {
    flexGrow: 1,
    alignItems: 'center',
  },
  topImage: {
    width: windowWidth,
    height: windowHeight * 0.25,
  },
  bottomImage: {
    width: windowWidth,
    height: windowHeight * 0.25,
  },
  registrationCenterBox: {
    width: '80%',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    marginTop: -windowHeight * 0.12, // Adjust this value to center the box vertically
    marginBottom: 50,
  },
  scrollViewContent: {
    flexGrow: 1,
  },
  signupTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
    color: '#125488',
  },
  inputContainer: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'black',
    marginBottom: 5,
  },
  input: {
    height: 40,
    width: '100%',
    padding: 10,
    borderWidth: 1,
    borderRadius: 5,
    borderColor: 'gray',
  },
  imagePickerContainer: {
    marginVertical: 10,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  photoButton: {
    backgroundColor: '#125488',
    padding: 10,
    borderRadius: 5,
    flex: 1,
    marginHorizontal: 5,
  },
  photoButtonText: {
    color: 'white',
    textAlign: 'center',
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignSelf: 'center',
    marginVertical: 10,
  },
  signupButton: {
    backgroundColor: '#125488',
    paddingHorizontal: 5,
    paddingVertical: 5,
    borderRadius: 29,
    marginTop: 10,
    width: windowWidth * 0.4,
    alignSelf: 'center',
  },
  signupButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  loginLinkContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 20,
  },
  alreadyText: {
    fontSize: 16,
    marginRight: 5,
  },
  loginText: {
    fontSize: 16,
    color: 'blue',
    textDecorationLine: 'underline',
  },
});

export default SignupScreen
