import React, {useState} from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ImageBackground, TextInput } from 'react-native';
import { Feather } from '@expo/vector-icons';
import ToggleSwitch from 'toggle-switch-react-native';
import Button from '../Components/button';
import GoBackButton from '../Components/goBackButton';
import { useNavigation, NavigationProp } from '@react-navigation/native';


interface SettingsScreenProps {
  navigation: any; 
}


const SettingsScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp<any>>();
  const handleSupportAndHelp = () => {
      navigation.navigate('SupportHelpScreen'); // Navigate to ProfilingScreen when signup button is pressed
    };
    const handleAccountSettings = () => {
      navigation.navigate('AccountSettingsScreen'); // Navigate to ProfilingScreen when signup button is pressed
    };

  const [feedback, setFeedback] = useState('');
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [silentModeEnabled, setsilentModeEnabled] = useState(false);
  const [LockScreennotificationsEnabled, setLockScreenNotificationsEnabled] = useState(true);

  const handleFeedbackSubmit = () => {
    // Logic to handle feedback submission
    console.log('Feedback submitted:', feedback);
  };
  return (
    <ImageBackground
      source={require('../screens/assets/images/backgroundIllustration.png')}
      style={{ flex: 1, backgroundColor: '#AFA58F', opacity: 0.62 }}
    >
      {/* Header */}
      <View style={styles.topHeader}>
       <GoBackButton/>
       <View style={styles.header}>
         {/* Gear icon */}
         <Feather name="settings" size={24} color="black" />
        <Text style={styles.headerText}>Settings</Text>
      </View>
      </View>

      {/* Notifications Card */}
      <Text style={styles.cardHeader}>Alert Settings</Text>
      <View style={styles.card}>
        <View style={styles.cardContent}>
        <Text style={styles.cardText}>Allow Notifications</Text>
        <ToggleSwitch
          isOn={notificationsEnabled}
          onColor="#47abea" // Blue color when "On"
          offColor="#e9c33e" // Yellow color when "Off"
          size="medium"
          onToggle={() => setNotificationsEnabled(!notificationsEnabled)}
        />
         </View>
      {/* Separator */}
      <View style={styles.separator} />

      {/* Silent Mode Card */}
      <View style={styles.cardContent}>
        <Text style={styles.cardText}>Silent Mode</Text>
        <ToggleSwitch
          isOn={silentModeEnabled}
          onColor="#47abea" // Blue color when "On"
          offColor="#e9c33e" // Yellow color when "Off"
          size="medium"
          onToggle={() => setsilentModeEnabled(!silentModeEnabled)}
        />
        </View>
      

      {/* Separator */}
      <View style={styles.separator} />

      {/* Alerts on LockScreen Card */}
      <View style={styles.cardContent}>
        <Text style={styles.cardText}>Alerts on LockScreen</Text>
        <ToggleSwitch
          isOn={LockScreennotificationsEnabled}
          onColor="#47abea" // Blue color when "On"
          offColor="#e9c33e" // Yellow color when "Off"
          size="medium"
          onToggle={() => setLockScreenNotificationsEnabled(!LockScreennotificationsEnabled)}
        />
        </View>
      </View>

      {/* Account Settings Card */}
      <TouchableOpacity style={styles.card} onPress={handleAccountSettings}>
        <Text style={styles.cardText}>Account Settings</Text>
      </TouchableOpacity>

      {/* Feedback Card */}
      <View style={styles.card}>
        <Text style={styles.cardText}>Feedback</Text>
        <TextInput
          multiline
          placeholder="Write your feedback here"
          onChangeText={() => {}}
          style={styles.feedbackInput}
        />
    
        <Button title="Send"
              onPress={handleFeedbackSubmit}
              buttonStyle={styles.sendButton}
              textStyle={styles.buttonText}/>
      </View>

      {/* Support and Help Card */}
      <TouchableOpacity style={styles.card} onPress={handleSupportAndHelp}>
        <Text style={styles.cardText}>Support and Help</Text>
      </TouchableOpacity>

      {/* SignOut Card */}
      <TouchableOpacity style={styles.card} onPress={() => {}}>
        <Text style={styles.cardText}>Sign Out</Text>
      </TouchableOpacity>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  topHeader:{
    flexDirection: 'row',
    marginTop: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 100,
  },
  headerText: {
    marginRight: 10,
    paddingLeft: 5,
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 18,
    fontWeight: 'bold',
  },
  cardHeader:{
    marginHorizontal: 13,
    fontSize: 16,
    paddingTop: 20,
    fontWeight: 'bold',
  },
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 8,
    marginHorizontal: 10,
    marginTop: 10,
    padding: 20,
  },
  cardContent:{
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 7,
  },
  cardText:{
    color: 'black',
    fontWeight: 'bold',
  },
  separator: {
    height: 1,
    backgroundColor: 'black',
    marginHorizontal: 10,
    marginVertical: 5,
  },
  feedbackInput: {
    backgroundColor: 'white',
    marginVertical: 10,
    padding: 5,
    borderRadius: 5,
  },
  sendButton: {
    backgroundColor: '#afa08f',
    padding: 10,
    borderRadius: 15,
    alignItems: 'center',
    marginLeft: '63%',
    justifyContent: 'center',
  },
  buttonText:{
    fontSize: 14,
    fontWeight: 'bold',
    color: '#000000',
  },
});

export default SettingsScreen;
