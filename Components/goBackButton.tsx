import React from 'react';
import { TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons'; // You can use any icon library

import { useNavigation } from '@react-navigation/native';

const GoBackButton: React.FC = () => {
  const navigation = useNavigation();

  const handleGoBack = () => {
    navigation.goBack();
  };

  return (
    <TouchableOpacity style={styles.container} onPress={handleGoBack}>
      <Ionicons name="chevron-back" size={24} color="black" />
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginLeft: 10, // Adjust spacing as needed
  },
});

export default GoBackButton;
