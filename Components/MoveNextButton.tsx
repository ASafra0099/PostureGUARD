// MoveNextButton.tsx
import React from 'react';
import { TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons'; // You can use any icon library
import { useNavigation } from '@react-navigation/native';

interface MoveNextButtonProps {
  targetScreen: string; // Name of the screen to navigate to
  onPress?: () => void; // Optional onPress function
}

const MoveNextButton: React.FC<MoveNextButtonProps> = ({ targetScreen, onPress }) => {
  const navigation = useNavigation();

  const handleMoveNext = () => {
    navigation.navigate(targetScreen as never); // Navigate to the specified screen
    if (onPress) {
      onPress(); // Call the onPress function if provided
    }
  };

  return (
    <TouchableOpacity style={styles.container} onPress={handleMoveNext}>
      <Ionicons name="chevron-forward" size={24} color="black" />
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginRight: 10, // Adjust spacing as needed
  },
});

export default MoveNextButton;
