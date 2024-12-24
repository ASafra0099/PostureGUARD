import React from 'react';
import { View, ViewStyle, StyleSheet, StyleProp } from 'react-native';

interface CardProps {
    width?: number;
    height?: number;
    backgroundColor?: string;
    style?: StyleProp<ViewStyle>; // Additional style
    children?: React.ReactNode; // Add children prop
  }
  

  const Card: React.FC<CardProps> = ({
    width = 200,
    height = 200,
    backgroundColor = '#FFFFFF',
    style,
    children, // Include children prop
  }) => {
    return (
      <View style={[styles.card, { width, height, backgroundColor }, style]}>
        {children} {/* Render children */}
      </View>
    );
  };
  

const styles = StyleSheet.create({
  card: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#CCCCCC',
    padding: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default Card;
