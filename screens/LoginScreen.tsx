import React from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity, Image, KeyboardAvoidingView, Dimensions, ScrollView } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation, NavigationProp } from '@react-navigation/native';

import Button from '../Components/button'; // Import the Button component

const LoginScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp<any>>();

  const handleLoginPress = () => {
    navigation.navigate('DashboardScreen');
  };

  const handleSignupLinkPress = () => {
    navigation.navigate('SignupScreen');
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
            <Text style={styles.signupTitle}>Login</Text>
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Email</Text>
              <TextInput placeholder="Email" style={styles.input} keyboardType="email-address" />
            </View>
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Password</Text>
              <TextInput placeholder="Password" style={styles.input} secureTextEntry />
            </View>
          </View>
          <Button
            title="Log In"
            onPress={handleLoginPress}
            buttonStyle={styles.loginButton}
            textStyle={styles.loginButtonText}
          />
          <View style={styles.signupLinkContainer}>
            <Text style={styles.createText}>Create new account?</Text>
            <TouchableOpacity onPress={handleSignupLinkPress}>
              <Text style={styles.signupText}>SignUp</Text>
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
  signupTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
    color: '#125488',
  },
  loginButton: {
    backgroundColor: '#125488',
    paddingHorizontal: 5,
    paddingVertical: 5,
    borderRadius: 29,
    marginTop: 10,
    width: 112,
    height: 34,
    alignSelf: 'center',
  },
  loginButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  signupLinkContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 20,
  },
  createText: {
    fontSize: 16,
    marginRight: 5,
  },
  signupText: {
    fontSize: 16,
    color: 'blue',
    textDecorationLine: 'underline',
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
});

export default LoginScreen;
