import React, { useState } from 'react';
import { View, Text, StyleSheet, ImageBackground, TouchableOpacity, TextInput } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import GoBackButton from '../Components/goBackButton';
import Button from '../Components/button';
import { LinearGradient } from 'expo-linear-gradient';

const AccountSettingsScreen: React.FC = () => {
  const [saveChanges, setsaveChanges] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
 

const [messageHeight, setMessageHeight] = useState(40); // Initial height of the text input
const [message, setMessage] = useState('');

const handleSaveChanges = () => {
    // Logic to handle sending feedback
    console.log('Changes saved:', saveChanges);
  };

  const handleSavePassword = () => {
    // Logic to handle sending feedback
    console.log('Changes saved:', confirmPassword);
  };

  

  return (
    <View style={styles.container}>
      <View style={styles.topHeader}>
      <GoBackButton/>
      <Text style={styles.heading}>Profile</Text>
      </View>

      {/* User Data Card */}
      <LinearGradient
        locations={[0.14, 1]}
        colors={["rgba(110, 153, 154, 0.80)", "rgba(124, 249, 219, 0.08)"]} style={styles.card}>
        {/* User Picture Circle */}
        <View style={styles.profilePicture}>
          {/* Add your user picture here */}
        </View>
        <View style={styles.userData}>
          <Text>Full Name: John Doe</Text>
          <Text>Age: 30</Text>
          <Text>Sex: Male</Text>
          </View>
        </LinearGradient>
      

      {/* Contact Details Card */}
      <LinearGradient
        locations={[0, 1]}
        colors={["rgba(110, 154, 153, 0.4)", "rgba(18, 84, 136, 0.4)"]} style={styles.card}>
        <Text style={styles.cardHeading}>Update Profile</Text>
  <View style={styles.inputRow}>
    <TextInput
      style={styles.textInput}
      placeholder="Enter your sex"
      onChangeText={(text) => {
        // Logic to update the user's sex
      }}
    />
  </View>
  <View style={styles.inputRow}>
    <TextInput
      style={styles.textInput}
      placeholder="Enter your age"
      keyboardType="numeric"
      onChangeText={(text) => {
        // Logic to update the user's age
      }}
    />
  </View>
  <Button title="Save Changes"
              onPress={handleSaveChanges }
              buttonStyle={styles.sendButton}
              textStyle={styles.buttonText}/>
      </LinearGradient>

      {/* App Rating Card */}
      <LinearGradient
        locations={[0, 1]}
        colors={[ "rgba(18, 84, 136, 0.4)" ,"rgba(175, 165, 143, 0.3)"]} style={styles.card}>
         <Text style={styles.cardHeading}>Password Settings</Text>
      <View style={styles.inputRow}>
        <TextInput
          secureTextEntry
          style={styles.textInput}
          value={password}
          onChangeText={setPassword}
          placeholder="Enter new password"
          placeholderTextColor="gray"
        />
      </View>
      <View style={styles.inputRow}>
        <TextInput
          secureTextEntry
          style={styles.textInput}
          value={confirmPassword}
          onChangeText={setConfirmPassword}
          placeholder="Confirm new password"
          placeholderTextColor="gray"
        />
      </View>
      {/* Add logic for handling password change and confirmation */}
      <Button title="Save Password"
              onPress={handleSavePassword }
              buttonStyle={styles.sendButton}
              textStyle={styles.buttonText}/>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#EAF0F2',
  },
  topHeader:{
    flexDirection: 'row',
    paddingBottom: 45,
    alignItems: 'flex-start',
    
  },
  heading: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 100,
    fontSize: 22,
    fontWeight: 'bold',
  },
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    borderRadius: 20,
    padding: 20,
    marginVertical: 5,
  },
  cardHeading: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 5,
    justifyContent: 'center',
    alignSelf: 'center',
  },
  cardText:{
    fontSize: 12,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  profilePicture: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'white',
    position: 'absolute',
    top: -50,
    left: '40%',
    borderWidth: 2,
    borderColor: 'black',
    overflow: 'hidden',
  },
  userData: {
    marginTop: 40,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },

  textInput: {
    borderBottomWidth: 1,
    borderBottomColor: 'black',
    flex: 1,
    paddingVertical: 5,
    fontSize: 16, // Input field font size
    color: 'black', // Input field text color
  },
  separator: {
    height: 1,
    backgroundColor: 'black',
    marginTop: 10,
    marginBottom: 10,
  },
  sendButton: {
    backgroundColor: '#afa08f',
    padding: 10,
    minWidth: 'auto',
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 6,
  },
  buttonText:{
    fontSize: 14,
    fontWeight: 'bold',
    color: '#000000',
  },
 
});

export default AccountSettingsScreen;
