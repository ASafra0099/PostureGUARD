import React, { useState } from 'react';
import { View, Text, StyleSheet, ImageBackground, TouchableOpacity, TextInput } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import GoBackButton from '../Components/goBackButton';
import Button from '../Components/button';
import { LinearGradient } from 'expo-linear-gradient';

const SupportHelpScreen: React.FC = () => {
  const [feedback, setFeedback] = useState('');
  const [rating, setRating] = useState(0);

  const handleSendFeedback = () => {
    // Logic to handle sending feedback
    console.log('Feedback submitted:', feedback);
  };

  const handleStarPress = (starIndex: number) => {
    setRating(starIndex );
  };
  const handleRateUs = () => {
    // Logic to handle sending Rate Us
    console.log('Got Rated:', setRating);
  };
 

const [messageHeight, setMessageHeight] = useState(40); // Initial height of the text input
const [message, setMessage] = useState('');

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
        <Text style={styles.cardHeading}>Contact Details</Text>
        <Text style={styles.cardText}>Gmail:</Text>
        <Text>alishbaabdulsuboor@gmail.com</Text>
        <View style={styles.separator} />
        <TextInput
             multiline
             placeholder="Write your message here"
             onChangeText={(text) => {
             setMessage(text);
            // Calculate the new height based on the number of lines
             const numOfLines = text.split('\n').length;
             const newHeight = Math.min(80, numOfLines * 20); // Limit the maximum height to 120 pixels
             setMessageHeight(newHeight); }} style={[styles.textInput, { height: messageHeight }]}/>

        <Button title="Send"
              onPress={handleSendFeedback}
              buttonStyle={styles.sendButton}
              textStyle={styles.buttonText}/>
      </LinearGradient>

      {/* App Rating Card */}
      <LinearGradient
        locations={[0, 1]}
        colors={[ "rgba(18, 84, 136, 0.4)" ,"rgba(175, 165, 143, 0.3)"]} style={styles.card}>
        <Text style={styles.cardHeading}>Rate Us</Text>
        <View style={styles.starContainer}>
          {[1, 2, 3, 4, 5].map((index) => (
            <TouchableOpacity key={index} onPress={() => handleStarPress(index)} style={styles.star}>
              <AntDesign name={index <= rating ? 'star' : 'staro'} size={24} color={index <= rating ? 'yellow' : 'white'} />
            </TouchableOpacity>
          ))}
        </View>
        <Button title="Rate Us"
              onPress={handleRateUs }
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
    marginVertical: 10,
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
  input: {
    backgroundColor: 'white',
    borderRadius: 5,
    padding: 10,
    marginTop: 10,
  },
  textInput: {
    borderWidth: 1,
    borderColor: 'gray',
    borderRadius: 5,
    marginTop: 5,
    padding: 10,
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
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  buttonText:{
    fontSize: 14,
    fontWeight: 'bold',
    color: '#000000',
  },
  starContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 10,
  },
  star: {
    marginHorizontal: 5,
  },
  rateButton: {
    backgroundColor: '#007bff',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
    alignItems: 'center',
  },
  rateButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
});

export default SupportHelpScreen;
