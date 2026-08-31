CREATE DATABASE ecommerce_returns;
USE ecommerce_returns;
CREATE TABLE returns_data (
    Order_ID VARCHAR(20),
    Product_ID VARCHAR(20),
    User_ID VARCHAR(20),
    Order_Date DATE,
    Product_Category VARCHAR(100),
    Product_Price DECIMAL(10,2),
    Order_Quantity INT,
    Discount_Applied DECIMAL(10,2),
    Shipping_Method VARCHAR(50),
    Payment_Method VARCHAR(50),
    User_Age INT,
    User_Gender VARCHAR(20),
    User_Location VARCHAR(100),
    Return_Status VARCHAR(30),
    Return_Reason VARCHAR(100),
    Days_to_Return INT,
    Order_Value DECIMAL(12,2),
    Return_Cost DECIMAL(12,2),
    Profit_Loss DECIMAL(12,2),
    CO2_Emissions DECIMAL(10,2),
    Packaging_Waste DECIMAL(10,2),
    CO2_Saved DECIMAL(10,2),
    Waste_Avoided DECIMAL(10,2)
);

SELECT COUNT(*)
FROM returns_data;

SELECT COUNT(*) AS total_records
FROM returns_data;

SELECT *
FROM returns_data
LIMIT 10;