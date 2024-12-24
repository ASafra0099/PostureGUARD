import React, {useEffect, useState} from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ImageBackground, TextInput } from 'react-native';
import ToggleSwitch from 'toggle-switch-react-native';
import Button from '../Components/button';
import GoBackButton from '../Components/goBackButton';
import { useNavigation, NavigationProp } from '@react-navigation/native';
import Checkbox from '../Components/Checkbox';
import RefreshButton from '../Components/RefreshButton';
import { Ionicons } from '@expo/vector-icons';

interface BindingScreen2Props {
  navigation: any; 
}

const BindingScreen2: React.FC<BindingScreen2Props> = () => {
  const navigation = useNavigation<NavigationProp<any>>();
  const [isChecked, setIsChecked] = useState(false);

  const handleCheckboxPress = () => {
    setIsChecked(!isChecked);
  };
const [isdevConnected, setDeviceConnected] = useState(true);
const handleRefresh = () => {
  // Handle refresh action here
  console.log('Refreshing...');
};


const [isConnected, setIsConnected] = useState(false); // Assuming initially not connected

// Example useEffect to update connection status
useEffect(() => {
  // Your logic to check device connection status
  // For example, you might use some asynchronous function to check device connection
  // For now, let's assume it changes the isConnected state after some time
  const checkConnection = async () => {
    // Simulating some asynchronous operation
    await new Promise(resolve => setTimeout(resolve, 2000)); // Simulating 2 seconds delay

    // Example: Setting isConnected to true if connected, false otherwise
    setIsConnected(/* Your logic to determine connection status */ false);
  };

  checkConnection(); // Trigger connection check
}, []); // Empty dependency array ensures this effect runs only once, similar to componentDidMount


return (
 <View style={styles.container}>
    {/* Header */}
    <View style={styles.topHeader}>
     <GoBackButton/> 
      <Text style={styles.headerText}>Add Device</Text>
    </View>

    {/* Notifications Card */}
    <View style={styles.firstCardHead}>
    <Text style={styles.cardHeader}>CONNECTED DEVICE</Text>
    <View style={styles.toggleBtn}>
    <ToggleSwitch
        isOn={isdevConnected}
        onColor="#47abea" // Blue color when "On"
        offColor="#e9c33e" // Yellow color when "Off"
        size="medium"
        onToggle={() => setDeviceConnected(!isdevConnected)}
      />
      </View>
      </View>
    <View style={styles.card1}>
    <Text style={styles.cardText}>BlinkSync </Text>
    <View style={styles.checkBox}>
        <Checkbox checked={isChecked} onPress={handleCheckboxPress} />
        </View>
    </View>
    {/* second card container */}
    <View style={styles.firstCardHead}>
        <Text style={styles.cardHeader}>MY DEVICES</Text>
        <View style={styles.refreshBtn}>
        <RefreshButton onPress={handleRefresh} />
        </View>
      </View>
      <View style={styles.card}>
        <View style={styles.cardContent}>
        <Text style={styles.cardText}>FocusMate</Text>
        <Text style={styles.connectionText}>{isConnected ? 'Connected' : 'Not Connected'}</Text>
         </View>
      {/* Separator */}
      <View style={styles.separator} />

      {/* Silent Mode Card */}
      <View style={styles.cardContent}>
        <Text style={styles.cardText}>CamSync Pro</Text>
        <Text style={styles.connectionText}>{isConnected ? 'Connected' : 'Not Connected'}</Text>
        </View>
      

      {/* Separator */}
      <View style={styles.separator} />

      {/* Alerts on LockScreen Card */}
      <View style={styles.cardContent}>
        <Text style={styles.cardText}>LensLinker</Text>
        <Text style={styles.connectionText}>{isConnected ? 'Connected' : 'Not Connected'}</Text>
        </View>
         {/* Separator */}
      <View style={styles.separator} />
      </View>
      
      {/* third card container */}
      <View style={styles.firstCardHead}>
        <Text style={styles.cardHeader}>OTHER DEVICES</Text>
        <View style={styles.refreshBtn}>
        <RefreshButton onPress={handleRefresh} />
        </View>
      </View>
      <View style={styles.card}>
        <View style={styles.cardContent}>
        <Text style={styles.cardText}>SnapLinker</Text>
        
         </View>
      {/* Separator */}
      <View style={styles.separator} />

      {/* Silent Mode Card */}
      <View style={styles.cardContent}>
        <Text style={styles.cardText}>BT5</Text>
        
        </View>
      

      {/* Separator */}
      <View style={styles.separator} />

      {/* Alerts on LockScreen Card */}
      <View style={styles.cardContent}>
        <Text style={styles.cardText}>PixelPulse</Text>
        
        </View>
         {/* Separator */}
      <View style={styles.separator} />
      </View>
      


  </View>
);
};

const styles = StyleSheet.create({
container:{
  flex: 1,
  paddingTop: 20,
  backgroundColor: 'rgba(175, 165, 143, 0.62)',
},
  topHeader:{
  flexDirection: 'row',
justifyContent: 'flex-start',
},
headerText: {
  marginRight: 10,
  marginLeft: 100,
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: 18,
  fontWeight: 'bold',
},
firstCardHead:{
  flexDirection: 'row',
  justifyContent: 'space-around',

},
cardHeader:{
  fontSize: 16,
  paddingTop: 20,
  fontWeight: 'bold',
},
toggleBtn:{
  top: 20,
  paddingLeft: 90,
},
checkBox:{
  paddingLeft: 200,
},
refreshBtn:{
  top: 10,
  paddingLeft: 150,
},
card1: {
  flexDirection: 'row',
  backgroundColor: 'rgba(255, 255, 255, 0.5)',
  borderRadius: 8,
  marginHorizontal: 10,
  marginTop: 10,
  padding: 20,
  shadowOffset: { width: 3, height: 3 },
  shadowOpacity: 0.7,
  shadowRadius: 4,
  elevation: 2,
},
card: {
  backgroundColor: 'rgba(255, 255, 255, 0.5)',
  borderRadius: 8,
  marginHorizontal: 10,
  marginTop: 10,
  padding: 20,
  shadowOffset: { width: 3, height: 3 },
  shadowOpacity: 0.7,
  shadowRadius: 4,
  elevation: 2,
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
connectionText: {
  marginRight: 8, // Adjust spacing as needed
  fontSize: 16,
  fontWeight: 'bold',
  color: '#47abea', // Color when connected
},
});
export default BindingScreen2;
