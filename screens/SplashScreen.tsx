import React, { useEffect } from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NavigationProp, RouteProp } from '@react-navigation/native';

interface Props {
  navigation: NavigationProp<any, any>;
}

const SplashScreen: React.FC<Props> = ({ navigation }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      // Navigate to the SignupScreen after 3 seconds
      navigation.navigate('PopupScreen');
    }, 3000); // 3000 milliseconds = 3 seconds

    // Clean up the timer to avoid memory leaks
    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <LinearGradient colors={['#6A9E99', '#125488']} style={styles.container}>
      <View>
        {/* Image */}
        <Image
          source={require('./assets/images/logoBox.png')} // Replace with the actual image path
          style={styles.image}
          resizeMode="contain"
        />
        <Text style={styles.title}>PostureGuard</Text>
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    ...StyleSheet.absoluteFillObject,
  },
  image: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '50%', // Adjust the size of the image as needed
    height: 'auto',
    aspectRatio: 2, // Ensure the image maintains its aspect ratio
  },
  title: {
    fontSize: 25,
    fontWeight: 'bold',
    color: '#fff', // White color
    marginTop: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default SplashScreen;
