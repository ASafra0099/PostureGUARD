import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

interface ButtonProps {
  title: string;
  onPress: () => void;
  buttonStyle?: object;
  textStyle?: object;
}

const Button: React.FC<ButtonProps> = ({ title, onPress, buttonStyle, textStyle }) => {
  return (
    <TouchableOpacity style={[styles.button, buttonStyle]} onPress={onPress}>
      <Text style={[styles.buttonText, textStyle]}>{title}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    backgroundColor: 'green', // Default button color
    paddingHorizontal: 10,
    paddingVertical: 10,
    borderRadius: 5,
    alignItems: 'center',
    width: 'auto',
    height: 'auto',
    justifyContent: 'center',
    alignSelf: 'center',
  },
  buttonText: {
    color: 'white', // Default text color
    fontSize: 14,
    fontWeight: 'bold',
    
  },
});

export default Button;
