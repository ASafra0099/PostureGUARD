import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { AntDesign } from '@expo/vector-icons'; // Import icons from Expo
import { useNavigation, NavigationProp } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';

const BottomMenu: React.FC = () => {
  const navigation = useNavigation<NavigationProp<any>>();

  const [activeMenu, setActiveMenu] = useState('DashboardScreen'); // Track active menu

  const handleMenuPress = (menu: string) => {
    setActiveMenu(menu);
    navigation.navigate(menu);
  };

  return (
    <LinearGradient colors={['#6A9E99', '#125488']} style={styles.container}>
      <TouchableOpacity
        style={styles.menuItem}
        onPress={() => handleMenuPress('DashboardScreen')}>
        <View style={[styles.iconContainer, activeMenu === 'DashboardScreen' && styles.activeMenuItem]}>
          <AntDesign name="clockcircle" size={24} color="white" />
        </View>
        <Text style={styles.menuText}>Real Time</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.menuItem}
        onPress={() => handleMenuPress('PracticeScreen')}>
        <View style={[styles.iconContainer, activeMenu === 'PracticeScreen' && styles.activeMenuItem]}>
          <AntDesign name="book" size={24} color="white" />
        </View>
        <Text style={styles.menuText}>Practice</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.menuItem}
        onPress={() => handleMenuPress('AnalysisScreen')}>
        <View style={[styles.iconContainer, activeMenu === 'AnalysisScreen' && styles.activeMenuItem]}>
          <AntDesign name="linechart" size={24} color="white" />
        </View>
        <Text style={styles.menuText}>Analysis</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.menuItem}
        onPress={() => handleMenuPress('SettingsScreen')}>
        <View style={[styles.iconContainer, activeMenu === 'SettingsScreen' && styles.activeMenuItem]}>
          <AntDesign name="setting" size={24} color="white" />
        </View>
        <Text style={styles.menuText}>Settings</Text>
      </TouchableOpacity>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingVertical: 5,
    width: '100%',
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    borderBottomLeftRadius: 0, // Set to 0 if you want only the top corners rounded
    borderBottomRightRadius: 0,
  },
  menuItem: {
    alignItems: 'center',
  },
  iconContainer: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  menuText: {
    fontSize: 12,
    marginTop: 5,
    color: 'white',
  },
  activeMenuItem: {
    backgroundColor: '#AFA58F', // Background color for the active menu item
    borderRadius: 20, // Half of the width/height to make it circular
  },
});

export default BottomMenu;
