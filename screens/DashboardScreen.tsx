import React from 'react';
import { View, Text, StyleSheet, Image, ScrollView } from 'react-native';
import Button from '../Components/button';
import { useNavigation, NavigationProp } from '@react-navigation/native';
import BottomMenu from '../Components/bottomMenu';

interface DashboardScreenProps {
  navigation: any; 
}

const DashboardScreen: React.FC<DashboardScreenProps> = () => {
    const navigation = useNavigation<NavigationProp<any>>();
    const handleBindCamPress = () => {
        navigation.navigate('BindingScreen1'); // Navigate to ProfilingScreen when signup button is pressed
      };

    return (
      <View style={styles.container}>
        {/* User Account Details Section */}
        <View style={styles.header}>
        <Text style={styles.headerText}>
          <Text style={styles.postureText}>Posture</Text>
          <Text style={styles.guardText}>Guard</Text>
        </Text>
      </View>
          <View style={styles.userDetailsHighlight}>
          <View style={styles.userCard}>
            <View style={styles.userDetailsContainer}>
            <Text style={styles.userInfoText}>User Name</Text>
            <Text style={styles.userInfoText}>Age: XX</Text>
            <Text style={styles.userInfoText}>Sex: XX</Text>
            </View>
            <Image
            source={require('./assets/images/barbell.png')}
            style={styles.profileImage}
          />
          </View>
          
        </View>
        {/* Cards Section */}
        <ScrollView contentContainerStyle={styles.cardsContainer}>
          {/* Camera Connection Card */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Camera Connection</Text>
            <Image
              source={require('./assets/images/triangle.png')}
              style={styles.cardIcon}
            />
             <Button 
              title="Bind Camera" 
              onPress={handleBindCamPress} 
              buttonStyle={styles.buttonStyle} 
              textStyle={styles.buttonText}
            />
          </View>
          {/* Score Graph Card */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Score Graph</Text>
            <Image
              source={require('./assets/images/triangle.png')}
              style={styles.cardIcon}
            />
          </View>
          {/* Monitoring Card */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Monitoring</Text>
            <Image
              source={require('./assets/images/triangle.png')}
              style={styles.cardIcon}
            />
           
          </View>
        </ScrollView>
        {/* Bottom Menu */}
        <BottomMenu />
      </View>
    );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'rgba(175, 165, 143, 0.55)',
    alignItems: 'center',
    
  },
  header: {
    alignItems: 'center',
    width: '100%',
    height: '10%',
    backgroundColor: 'rgba(175, 165, 143, 0.55)',
  },
  headerText: {
    fontSize: 24,
    top: 20,
    fontWeight: 'bold',
    color: '#2E86C1', // Color for "Posture"
  },
  postureText: {
    color: '#2E86C1', // Color for "Posture"
  },
  guardText: {
    color: '#58D68D', // Color for "Guard"
  },
  userDetailsContainer: {
    flexDirection: 'column',
    alignItems: 'center',
   
    
  },
  userCard:{
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  userDetailsHighlight: {
    padding: 20,
    width: '100%',
    backgroundColor: 'rgba(173, 217, 216, 0.4)',
  },
 
  userInfoText: {
    fontSize: 16,
    marginBottom: 5,
    color: '#2E86C1',
    paddingLeft: 20,
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderColor: '#2E86C1',
    borderWidth: 1,
    marginLeft: 100,
  },
  cardsContainer: {
    alignItems: 'center',
    bottom: 10,
  },
  card: {
    backgroundColor: 'rgba(175, 165, 143, 0.25)',
    borderRadius: 10,
    padding: 20,
    marginBottom: 20,
    width: '100%',
    alignItems: 'center',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#2E86C1',
  },
  cardIcon: {
    width: 50,
    height: 50,
    marginBottom: 10,
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
  
export default DashboardScreen;
