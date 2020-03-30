from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol

class ProductSummary(MRJob):
    
    OUTPUT_PROTOCOL = RawValueProtocol

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer)
        ]

    def mapper(self, _, line):
        #CSV Header: [Customer ID, Transaction ID, Date, Product ID, Item Cost]
        entries = line.split(",")
        custID = str(entries[0])
        productID = str(entries[3])
        itemCost = float(entries[4])
        
        yield productID, (itemCost, custID)
 
    def reducer(self, productID, values):
        #OUTPUT: [Product ID, Customer Count, Total Revenue]
        customerCount = set()
        totalRevenue = 0.00
        for val in values:
            customerCount.add(val[1])
            totalRevenue += val[0]
        result = str(productID) + "," + str(len(customerCount)) + ","+str(format(totalRevenue, '.2f'))
        yield None, result

if __name__ == '__main__':
    ProductSummary.run()