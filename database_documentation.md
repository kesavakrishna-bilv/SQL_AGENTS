## Table: DatabaseLog
| Column Name | Data Type |
|-------------|-----------|
| DatabaseLogID | int |
| PostTime | datetime |
| DatabaseUser | nvarchar |
| Event | nvarchar |
| Schema | nvarchar |
| Object | nvarchar |
| TSQL | nvarchar |
| XmlEvent | xml |

## Table: AdventureWorksDWBuildVersion
| Column Name | Data Type |
|-------------|-----------|
| DBVersion | nvarchar |
| VersionDate | datetime |

## Table: DimAccount
| Column Name | Data Type |
|-------------|-----------|
| AccountKey | int |
| ParentAccountKey | int |
| AccountCodeAlternateKey | int |
| ParentAccountCodeAlternateKey | int |
| AccountDescription | nvarchar |
| AccountType | nvarchar |
| Operator | nvarchar |
| CustomMembers | nvarchar |
| ValueType | nvarchar |
| CustomMemberOptions | nvarchar |

## Table: DimCurrency
| Column Name | Data Type |
|-------------|-----------|
| CurrencyKey | int |
| CurrencyAlternateKey | nchar |
| CurrencyName | nvarchar |

## Table: DimCustomer
| Column Name | Data Type |
|-------------|-----------|
| CustomerKey | int |
| GeographyKey | int |
| CustomerAlternateKey | nvarchar |
| Title | nvarchar |
| FirstName | nvarchar |
| MiddleName | nvarchar |
| LastName | nvarchar |
| NameStyle | bit |
| BirthDate | date |
| MaritalStatus | nchar |
| Suffix | nvarchar |
| Gender | nvarchar |
| EmailAddress | nvarchar |
| YearlyIncome | money |
| TotalChildren | tinyint |
| NumberChildrenAtHome | tinyint |
| EnglishEducation | nvarchar |
| SpanishEducation | nvarchar |
| FrenchEducation | nvarchar |
| EnglishOccupation | nvarchar |
| SpanishOccupation | nvarchar |
| FrenchOccupation | nvarchar |
| HouseOwnerFlag | nchar |
| NumberCarsOwned | tinyint |
| AddressLine1 | nvarchar |
| AddressLine2 | nvarchar |
| Phone | nvarchar |
| DateFirstPurchase | date |
| CommuteDistance | nvarchar |

## Table: DimDate
| Column Name | Data Type |
|-------------|-----------|
| DateKey | int |
| FullDateAlternateKey | date |
| DayNumberOfWeek | tinyint |
| EnglishDayNameOfWeek | nvarchar |
| SpanishDayNameOfWeek | nvarchar |
| FrenchDayNameOfWeek | nvarchar |
| DayNumberOfMonth | tinyint |
| DayNumberOfYear | smallint |
| WeekNumberOfYear | tinyint |
| EnglishMonthName | nvarchar |
| SpanishMonthName | nvarchar |
| FrenchMonthName | nvarchar |
| MonthNumberOfYear | tinyint |
| CalendarQuarter | tinyint |
| CalendarYear | smallint |
| CalendarSemester | tinyint |
| FiscalQuarter | tinyint |
| FiscalYear | smallint |
| FiscalSemester | tinyint |

## Table: DimDepartmentGroup
| Column Name | Data Type |
|-------------|-----------|
| DepartmentGroupKey | int |
| ParentDepartmentGroupKey | int |
| DepartmentGroupName | nvarchar |

## Table: DimEmployee
| Column Name | Data Type |
|-------------|-----------|
| EmployeeKey | int |
| ParentEmployeeKey | int |
| EmployeeNationalIDAlternateKey | nvarchar |
| ParentEmployeeNationalIDAlternateKey | nvarchar |
| SalesTerritoryKey | int |
| FirstName | nvarchar |
| LastName | nvarchar |
| MiddleName | nvarchar |
| NameStyle | bit |
| Title | nvarchar |
| HireDate | date |
| BirthDate | date |
| LoginID | nvarchar |
| EmailAddress | nvarchar |
| Phone | nvarchar |
| MaritalStatus | nchar |
| EmergencyContactName | nvarchar |
| EmergencyContactPhone | nvarchar |
| SalariedFlag | bit |
| Gender | nchar |
| PayFrequency | tinyint |
| BaseRate | money |
| VacationHours | smallint |
| SickLeaveHours | smallint |
| CurrentFlag | bit |
| SalesPersonFlag | bit |
| DepartmentName | nvarchar |
| StartDate | date |
| EndDate | date |
| Status | nvarchar |
| EmployeePhoto | varbinary |

## Table: DimGeography
| Column Name | Data Type |
|-------------|-----------|
| GeographyKey | int |
| City | nvarchar |
| StateProvinceCode | nvarchar |
| StateProvinceName | nvarchar |
| CountryRegionCode | nvarchar |
| EnglishCountryRegionName | nvarchar |
| SpanishCountryRegionName | nvarchar |
| FrenchCountryRegionName | nvarchar |
| PostalCode | nvarchar |
| SalesTerritoryKey | int |
| IpAddressLocator | nvarchar |

## Table: DimOrganization
| Column Name | Data Type |
|-------------|-----------|
| OrganizationKey | int |
| ParentOrganizationKey | int |
| PercentageOfOwnership | nvarchar |
| OrganizationName | nvarchar |
| CurrencyKey | int |

## Table: DimProduct
| Column Name | Data Type |
|-------------|-----------|
| ProductKey | int |
| ProductAlternateKey | nvarchar |
| ProductSubcategoryKey | int |
| WeightUnitMeasureCode | nchar |
| SizeUnitMeasureCode | nchar |
| EnglishProductName | nvarchar |
| SpanishProductName | nvarchar |
| FrenchProductName | nvarchar |
| StandardCost | money |
| FinishedGoodsFlag | bit |
| Color | nvarchar |
| SafetyStockLevel | smallint |
| ReorderPoint | smallint |
| ListPrice | money |
| Size | nvarchar |
| SizeRange | nvarchar |
| Weight | float |
| DaysToManufacture | int |
| ProductLine | nchar |
| DealerPrice | money |
| Class | nchar |
| Style | nchar |
| ModelName | nvarchar |
| LargePhoto | varbinary |
| EnglishDescription | nvarchar |
| FrenchDescription | nvarchar |
| ChineseDescription | nvarchar |
| ArabicDescription | nvarchar |
| HebrewDescription | nvarchar |
| ThaiDescription | nvarchar |
| GermanDescription | nvarchar |
| JapaneseDescription | nvarchar |
| TurkishDescription | nvarchar |
| StartDate | datetime |
| EndDate | datetime |
| Status | nvarchar |

## Table: DimProductCategory
| Column Name | Data Type |
|-------------|-----------|
| ProductCategoryKey | int |
| ProductCategoryAlternateKey | int |
| EnglishProductCategoryName | nvarchar |
| SpanishProductCategoryName | nvarchar |
| FrenchProductCategoryName | nvarchar |

## Table: DimProductSubcategory
| Column Name | Data Type |
|-------------|-----------|
| ProductSubcategoryKey | int |
| ProductSubcategoryAlternateKey | int |
| EnglishProductSubcategoryName | nvarchar |
| SpanishProductSubcategoryName | nvarchar |
| FrenchProductSubcategoryName | nvarchar |
| ProductCategoryKey | int |

## Table: DimPromotion
| Column Name | Data Type |
|-------------|-----------|
| PromotionKey | int |
| PromotionAlternateKey | int |
| EnglishPromotionName | nvarchar |
| SpanishPromotionName | nvarchar |
| FrenchPromotionName | nvarchar |
| DiscountPct | float |
| EnglishPromotionType | nvarchar |
| SpanishPromotionType | nvarchar |
| FrenchPromotionType | nvarchar |
| EnglishPromotionCategory | nvarchar |
| SpanishPromotionCategory | nvarchar |
| FrenchPromotionCategory | nvarchar |
| StartDate | datetime |
| EndDate | datetime |
| MinQty | int |
| MaxQty | int |

## Table: DimReseller
| Column Name | Data Type |
|-------------|-----------|
| ResellerKey | int |
| GeographyKey | int |
| ResellerAlternateKey | nvarchar |
| Phone | nvarchar |
| BusinessType | varchar |
| ResellerName | nvarchar |
| NumberEmployees | int |
| OrderFrequency | char |
| OrderMonth | tinyint |
| FirstOrderYear | int |
| LastOrderYear | int |
| ProductLine | nvarchar |
| AddressLine1 | nvarchar |
| AddressLine2 | nvarchar |
| AnnualSales | money |
| BankName | nvarchar |
| MinPaymentType | tinyint |
| MinPaymentAmount | money |
| AnnualRevenue | money |
| YearOpened | int |

## Table: DimSalesReason
| Column Name | Data Type |
|-------------|-----------|
| SalesReasonKey | int |
| SalesReasonAlternateKey | int |
| SalesReasonName | nvarchar |
| SalesReasonReasonType | nvarchar |

## Table: DimSalesTerritory
| Column Name | Data Type |
|-------------|-----------|
| SalesTerritoryKey | int |
| SalesTerritoryAlternateKey | int |
| SalesTerritoryRegion | nvarchar |
| SalesTerritoryCountry | nvarchar |
| SalesTerritoryGroup | nvarchar |
| SalesTerritoryImage | varbinary |

## Table: DimScenario
| Column Name | Data Type |
|-------------|-----------|
| ScenarioKey | int |
| ScenarioName | nvarchar |

## Table: FactAdditionalInternationalProductDescription
| Column Name | Data Type |
|-------------|-----------|
| ProductKey | int |
| CultureName | nvarchar |
| ProductDescription | nvarchar |

## Table: FactCallCenter
| Column Name | Data Type |
|-------------|-----------|
| FactCallCenterID | int |
| DateKey | int |
| WageType | nvarchar |
| Shift | nvarchar |
| LevelOneOperators | smallint |
| LevelTwoOperators | smallint |
| TotalOperators | smallint |
| Calls | int |
| AutomaticResponses | int |
| Orders | int |
| IssuesRaised | smallint |
| AverageTimePerIssue | smallint |
| ServiceGrade | float |
| Date | datetime |

## Table: FactCurrencyRate
| Column Name | Data Type |
|-------------|-----------|
| CurrencyKey | int |
| DateKey | int |
| AverageRate | float |
| EndOfDayRate | float |
| Date | datetime |

## Table: FactFinance
| Column Name | Data Type |
|-------------|-----------|
| FinanceKey | int |
| DateKey | int |
| OrganizationKey | int |
| DepartmentGroupKey | int |
| ScenarioKey | int |
| AccountKey | int |
| Amount | float |
| Date | datetime |

## Table: FactInternetSales
| Column Name | Data Type |
|-------------|-----------|
| ProductKey | int |
| OrderDateKey | int |
| DueDateKey | int |
| ShipDateKey | int |
| CustomerKey | int |
| PromotionKey | int |
| CurrencyKey | int |
| SalesTerritoryKey | int |
| SalesOrderNumber | nvarchar |
| SalesOrderLineNumber | tinyint |
| RevisionNumber | tinyint |
| OrderQuantity | smallint |
| UnitPrice | money |
| ExtendedAmount | money |
| UnitPriceDiscountPct | float |
| DiscountAmount | float |
| ProductStandardCost | money |
| TotalProductCost | money |
| SalesAmount | money |
| TaxAmt | money |
| Freight | money |
| CarrierTrackingNumber | nvarchar |
| CustomerPONumber | nvarchar |
| OrderDate | datetime |
| DueDate | datetime |
| ShipDate | datetime |

## Table: FactInternetSalesReason
| Column Name | Data Type |
|-------------|-----------|
| SalesOrderNumber | nvarchar |
| SalesOrderLineNumber | tinyint |
| SalesReasonKey | int |

## Table: FactProductInventory
| Column Name | Data Type |
|-------------|-----------|
| ProductKey | int |
| DateKey | int |
| MovementDate | date |
| UnitCost | money |
| UnitsIn | int |
| UnitsOut | int |
| UnitsBalance | int |

## Table: FactResellerSales
| Column Name | Data Type |
|-------------|-----------|
| ProductKey | int |
| OrderDateKey | int |
| DueDateKey | int |
| ShipDateKey | int |
| ResellerKey | int |
| EmployeeKey | int |
| PromotionKey | int |
| CurrencyKey | int |
| SalesTerritoryKey | int |
| SalesOrderNumber | nvarchar |
| SalesOrderLineNumber | tinyint |
| RevisionNumber | tinyint |
| OrderQuantity | smallint |
| UnitPrice | money |
| ExtendedAmount | money |
| UnitPriceDiscountPct | float |
| DiscountAmount | float |
| ProductStandardCost | money |
| TotalProductCost | money |
| SalesAmount | money |
| TaxAmt | money |
| Freight | money |
| CarrierTrackingNumber | nvarchar |
| CustomerPONumber | nvarchar |
| OrderDate | datetime |
| DueDate | datetime |
| ShipDate | datetime |

## Table: FactSalesQuota
| Column Name | Data Type |
|-------------|-----------|
| SalesQuotaKey | int |
| EmployeeKey | int |
| DateKey | int |
| CalendarYear | smallint |
| CalendarQuarter | tinyint |
| SalesAmountQuota | money |
| Date | datetime |

## Table: FactSurveyResponse
| Column Name | Data Type |
|-------------|-----------|
| SurveyResponseKey | int |
| DateKey | int |
| CustomerKey | int |
| ProductCategoryKey | int |
| EnglishProductCategoryName | nvarchar |
| ProductSubcategoryKey | int |
| EnglishProductSubcategoryName | nvarchar |
| Date | datetime |

## Table: NewFactCurrencyRate
| Column Name | Data Type |
|-------------|-----------|
| AverageRate | real |
| CurrencyID | nvarchar |
| CurrencyDate | date |
| EndOfDayRate | real |
| CurrencyKey | int |
| DateKey | int |

## Table: ProspectiveBuyer
| Column Name | Data Type |
|-------------|-----------|
| ProspectiveBuyerKey | int |
| ProspectAlternateKey | nvarchar |
| FirstName | nvarchar |
| MiddleName | nvarchar |
| LastName | nvarchar |
| BirthDate | datetime |
| MaritalStatus | nchar |
| Gender | nvarchar |
| EmailAddress | nvarchar |
| YearlyIncome | money |
| TotalChildren | tinyint |
| NumberChildrenAtHome | tinyint |
| Education | nvarchar |
| Occupation | nvarchar |
| HouseOwnerFlag | nchar |
| NumberCarsOwned | tinyint |
| AddressLine1 | nvarchar |
| AddressLine2 | nvarchar |
| City | nvarchar |
| StateProvinceCode | nvarchar |
| PostalCode | nvarchar |
| Phone | nvarchar |
| Salutation | nvarchar |
| Unknown | int |

## Table: sysdiagrams
| Column Name | Data Type |
|-------------|-----------|
| name | nvarchar |
| principal_id | int |
| diagram_id | int |
| version | int |
| definition | varbinary |

