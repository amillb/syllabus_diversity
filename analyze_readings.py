#!/usr/bin/python
# -*- coding: utf-8 -*

"""
Estimate race and gender for readings in course syllabi
For details, see:
Millard-Ball, Adam; Desai, Garima; and Fahrney, Jessica (2024). "Diversifying Planning Education through Course Readings." 
Journal of Planning Education and Research, 44(2): 527-534.
https://doi.org/10.1177/0739456X211001936

December 2020
"""

import os, sys
if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, requires Python 3. You are running Python 2.\n")
    sys.exit(1)

import numpy as np
import pandas as pd
import re
import ethnicolr
import gender_guesser.detector as gender

class syllabusAnalyzer():
    def  __init__(self, inFn, outFn):
        self.inFn = inFn
        self.outFn = outFn
        self.df = None      # populated in loadData()
        self.outDf = None   # populated in addRaceGender()

    def loadData(self, fileobject=None):
        """Loads data on each reading into a dataframe of names
           If fileobject is None, data is read from file with self.inFn
           Otherwise, read from the file object"""
        if fileobject is None:
            if not(os.path.exists(self.inFn)):
                raise Exception('Cannot load input file {}. Does not exist.'.format(self.inFn))
            fileobject = self.inFn 

        if self.inFn.endswith('.xlsx') or self.inFn.endswith('.xls'):
            df = pd.read_excel(fileobject)
        elif self.inFn.endswith('.csv'):
            df = pd.read_csv(fileobject)
        else:
            raise Exception('Cannot load input file {}. Must be an .xlsx or .csv file'.format(self.inFn))

        df.columns = [str(cc).lower() for cc in df.columns]
        if 'reading' in df.columns:
            print('Loaded file in format 1: one row per reading')
            self.formattype = 1
            if 'readingid' not in df.columns:
                df['readingid'] = np.arange(1,len(df)+1)
        elif 'readingid' in df.columns and 'author' in df.columns:
            self.formattype = 2
            print('Loaded file in format 2: one row per author')
        else:
            raise Exception('Cannot load input file {}. Required columns not found'.format(self.inFn))

        if 'courseid' not in df.columns:
            df['courseid'] = 1

        self.df = df

    def parse_names(self, row):
        """extract names from the string, for format 1. Helper function for addRaceGender()"""
        # ** marks the names
        txt = row['reading'].split('**')[0]
        names = [ss.strip() for ss in txt.split(';')]
        return pd.DataFrame(names, index = [row['readingid']]*len(names), columns=['author'])

    def addRaceGender(self):
        """Creates a separate dataframe of names
        Adds the estimated race and gender to this dataframe
        Aggregates to readings and courseid"""
        
        if self.formattype == 1:
            nameDf = pd.concat([self.parse_names(row) for idx, row in self.df.iterrows()])
            nameDf.index.name = 'readingid'
            nameDf['courseid'] = self.df.set_index('readingid')[['courseid']]
            nameDf.set_index('courseid', append=True, inplace=True)
            nameDf = nameDf.swaplevel()
        else:
            nameDf = self.df[['readingid', 'author', 'courseid']].set_index(['courseid','readingid'])

        nameDf['lastname']  = nameDf.author.apply(lambda x: x.split(',')[0].strip().title())
        nameDf['firstname'] = nameDf.author.apply(lambda x: x.split(',')[1].strip().title().split()[0] if ',' in x else '')

        # add gender
        d = gender.Detector()
        nameDf['gender'] = nameDf.firstname.apply(lambda x: d.get_gender(x))
        nameDf['female'] = nameDf.gender.apply(lambda x: 1 if x in ['female','mostly_female'] else 0 if x in ['male','mostly_female'] else np.nan)
        
        
        # add race
        #ethnicolr can't handle a multiindex, so reset index and then readd it
        nameDf = ethnicolr.pred_fl_reg_name(nameDf.reset_index(),'lastname','firstname')
        nameDf.set_index(['courseid','readingid'], inplace=True)
        # change with v 0.8
        nameDf.rename(columns={cc+'_mean': cc.replace(cc+'_mean','') for cc in ['asian','hispanic','nh_black','nh_white']}, inplace=True)

        # Aggregate to article level

        if self.formattype==1:
            outDf = self.df.copy()
            outDf.set_index(['courseid','readingid'], inplace=True)
        else:
            outDf = pd.DataFrame(index = self.df.set_index(['courseid','readingid']).index.unique())

        outDf['N_authors'] = nameDf.groupby(level=[0,1]).size()
        outDf.fillna({'N_authors': 0}, inplace=True)
        outDf['pc_female'] = nameDf.groupby(level=[0,1]).female.mean()
        outDf = outDf.join(nameDf.groupby(level=[0,1])[['asian','hispanic','nh_black','nh_white']].mean())

        # for institutional authors, put NaN
        if self.formattype==1:
            mask = np.logical_not(outDf.reading.str.contains(r'\*\*'))
        else:
            nameDf['has_author'] = nameDf.author.str.contains(',')
            mask = nameDf.groupby(level=[0,1]).has_author.sum()==0

        for col in ['female','pc_female','asian','hispanic','nh_black','nh_white']:
            if col in outDf.columns:
                outDf.loc[mask, col] = np.nan
        outDf.loc[mask,'N_authors'] = 0
        
        self.outDf = outDf

    def outputResults(self):
        """Outputs the results, and optionally saves them"""
        if self.outDf.N_authors.sum() == 0:
            print('No authors detected. Please check your input file.')
            return

        courseids = self.outDf.index.get_level_values(0).unique()
        print('Results for input file: {}'.format(self.inFn))
        print('{} articles with {} authors from {} courses'.format(len(self.outDf), self.outDf.N_authors.sum(), len(courseids)))
        prettyNames = {'female':'female','pc_female':'female','asian':'Asian',
                'hispanic':'Hispanic/Latinx','nh_black':'non-Hispanic Black','nh_white':'non-Hispanic White'}
        
        for courseid in courseids:
            if len(courseids)>1: 
                print('\nResults for course id: {}'.format(courseid))
            for col in ['female','pc_female','asian','hispanic','nh_black','nh_white']:
                if col in self.outDf:
                    print('\tPercentage {}: {:.1f}%'.format(prettyNames[col], self.outDf.loc[courseid, col].mean()*100))

        if self.outFn is not None:
            if os.path.exists(self.outFn):
                ii = input('Output file {} already exists. Type Y to overwrite\n'.format(self.outFn)).lower()
                if ii not in ['y','yes']:
                    print('Skipping writing output file.')
                    return
            newNames = {'female':'frc_female','pc_female':'frc_female','asian':'pr_Asian',
                'hispanic':'pr_Hispanic','nh_black':'pr_non_Hispanic_Black','nh_white':'pr_non_Hispanic_White'}
            self.outDf.rename(columns=newNames, inplace=True)
            if self.outFn.endswith('.xlsx'):
                self.outDf.to_excel(self.outFn, index=True)
            elif self.outFn.endswith('.csv'):
                self.outDf.to_csv(self.outFn, index=True)
            else:
                self.outFn+='.csv'
                self.outDf.to_csv(self.outFn)
            print('Saved output file as {}'.format(self.outFn))

    def runall(self):
        self.loadData()
        self.addRaceGender()
        self.outputResults()

class jper_analysis():
    
    def  __init__(self):
        """This function was used for the analysis in the JPER paper
        It loads our data file and creates the plots for the published paper
        Most users should ignore it, but you might want to make use of some of the plotting functions"""

        path = '/Users/adammb/Documents/GDrive/Research/Syllabus diversity/Code and data/'
        self.inFn = path + 'ENVS 145 Course Revision Info.xlsx'
        self.facultyFn = path + 'Cites2019.xlsx'  # from Tom Sanchez
        self.figpath = '/Users/adammb/Documents/GDrive/Research/Syllabus diversity/Figures/'
        self.outFn = path+'readings_withracegender.csv'
        self.figsize = (6.5,3.5)
        self.figsize_big = (6.5,5.5)
        self.df = None

    def loadData(self):
        """Loads a pandas dataframe with the list of readings
        This function is heavily customized for the data source used in the JPER paper,
        and so is included in this class"""
        sheets = pd.read_excel(self.inFn, sheet_name=None)

        # get combined dataframe
        sheetnames = [ss for ss in sheets if ss!='Complete Syllabus List']
        for sheetname in sheetnames:
            sheets[sheetname]['Topic'] = sheetname
            
        df = pd.concat([sheets[ss][['Course ID', 'Reading','Topic']] for ss in sheetnames])
        df.rename(columns={'Reading':'reading', 'Course ID':'courseid'}, inplace=True)
        df['readingid'] = range(len(df))  # unique ID
        nonR1_courses = [3,4,13,14]
        df['R1'] = df.courseid.apply(lambda x: 0 if x in nonR1_courses else np.nan if x==33 else 1)
        landgrant = [9,12,19,20,21,25,34]
        df['landgrant'] = df.courseid.apply(lambda x: 1 if x in landgrant else np.nan if x==33 else 0)

        # recode some topics to have fewer groups
        relabels = [('LULUs, NIMBYs, YIMBYs','Land Use and Development'),
                    ('Energy','Energy, Water, Waste'), ('Waste','Energy, Water, Waste'),
                    ('Water','Energy, Water, Waste'), 
                    ('CEQA & NEPA','Measurement'),('Measuring a Green City', 'Measurement')]
        for oldl, newl in relabels:
            df.loc[df.Topic==oldl, 'Topic'] = newl

        sa = syllabusAnalyzer(None, None)
        sa.df,sa.formattype = df, 1
        sa.addRaceGender()
        sa.outDf.reset_index(level=0, inplace=True) # drop courseid from index
        self.df = sa.outDf
        self.addYear()

    def addYear(self):
        """Adds publication year to the dataframe
        Estimated based on first four-digit year found in the citation string"""
        self.df['year'] = self.df.reading.apply(lambda x: [xx for xx in re.findall(r'(\d{4})',x) if int(xx)>=1850 and int(xx)<=2020])
        assert self.df.year.apply(lambda x: isinstance(x, list)).all()
        self.df['year'] = self.df.year.apply(lambda x: np.nan if x==[] else int(x[0]) )
        self.df['yearbin'] = self.df.year.apply(lambda x: np.nan if pd.isnull(x) else 'Pre-1980' if x<1980 else '1980-89' if x<1990 else '1990-99' if x<2000 else '2000-09' if x<2010 else '2010-')

    def plotData(self):
        """Plots for paper"""
        if self.df is None:
            raise Exception('Need to load dataframe before running plotData')

        import matplotlib.pyplot as plt

        dfh = self.df[self.df.N_authors>0]
        fig, axes = plt.subplots(1,2)
        dfh.nh_white.hist(ax=axes[0])
        axes[0].set_xlabel('Probability non-Hispanic white')
        dfh.pc_female.hist(ax=axes[1])
        axes[1].set_xlabel('Fraction female')

        for ax in axes:
            ax.set_ylabel('Number of articles')
        plt.tight_layout()
        fig.savefig(self.figpath+'histograms.pdf')
        fig.savefig(self.figpath+'histograms.jpg', dpi=600)

        # distribution by course
        female = dfh.groupby('courseid').pc_female.mean().sort_values(ascending=False)        
        nonHispWhite = dfh.groupby('courseid').nh_white.mean().sort_values()    
        fig, axes = plt.subplots(1,2, figsize=self.figsize)
        female.plot(kind='bar', ax=axes[0])
        nonHispWhite.plot(kind='bar', ax=axes[1])
        axes[0].set_ylabel('Mean fraction of female authors')
        axes[1].set_ylabel('Mean fraction non-Hispanic White authors')
        for ax in axes:
            ax.set_xticks([])
            ax.set_xlabel('Course')
        plt.tight_layout()
        fig.savefig(self.figpath+'by_course.pdf')
        fig.savefig(self.figpath+'by_course.jpg', dpi=600)

        # boxplots by subject area
        fig, axes = plt.subplots(1,2, figsize=self.figsize_big)
        female_by_course = dfh.groupby(['courseid','Topic']).pc_female.mean()
        female_by_course.unstack().boxplot(ax=axes[0], rot=90)
        nhw_by_course = dfh.groupby(['courseid','Topic']).nh_white.mean()
        nhw_by_course.unstack().boxplot(ax=axes[1], rot=90)
        axes[0].set_ylabel('Mean fraction of female authors')
        axes[1].set_ylabel('Mean fraction non-Hispanic White authors')
        plt.tight_layout()
        fig.savefig(self.figpath+'by_topic.pdf')
        fig.savefig(self.figpath+'by_topic.jpg', dpi=600)

        # summary statistics
        outDf = dfh[['asian','hispanic', 'nh_black', 'nh_white','pc_female']].mean() 
        outDf.to_csv(self.figpath+'summarytable.csv')

        # trends by time, per reviewer request
        colOrder = ['Pre-1980', '1980-89', '1990-99', '2000-09', '2010-', ]
        fig, axes = plt.subplots(1,2, figsize=self.figsize)
        self.df.groupby('yearbin').pc_female.mean()[colOrder].plot(kind='bar', ax=axes[0])
        axes[0].set_ylabel('Mean fraction of female authors')

        self.df.groupby('yearbin').nh_white.mean()[colOrder].plot(kind='bar', ax=axes[1])
        axes[1].set_ylabel('Mean fraction non-Hispanic White authors')
        axes[0].set_xlabel('')
        axes[1].set_xlabel('')
        fig.tight_layout()

    def reportStats(self):
        """stats to quote in the text"""
        if self.df is None:
            raise Exception('Need to load dataframe before running reportStats')
        
        df = self.df.copy()
        print('Sample size: {} articles from {} courses'.format(len(df), len(df.courseid.unique())))

        # exclude institutional authors
        print('{} articles with 1+ human authors'.format((df.N_authors>0).sum()))

        # gender
        dfh = df[(df.N_authors>0) & pd.notnull(df.pc_female)]
        print('{:.0f}% female.'.format(dfh.pc_female.mean()*100))
        print('{:.0f}% male only. {:.0f}% female only.'.format(((dfh.pc_female==0).sum()/len(dfh)*100),((dfh.pc_female==1).sum()/len(dfh)*100)))
        dfs = dfh[dfh.N_authors==1]
        print('Single-authored articles: {:.0f}% male only. {:.0f}% female only.'.format(((dfs.pc_female==0).sum()/len(dfs)*100),((dfs.pc_female==1).sum()/len(dfs)*100)))

        # race and gender
        print('{:.0f}% where female POC.'.format(((dfh.pc_female>=0.5) & (dfh.nh_white<0.5)).sum()/len(dfh)*100))

        # race
        dfh = df[(df.N_authors>0)]
        assert  pd.isnull(dfh.nh_white).sum() == 0
        print('{:.0f}% > 90pc probability white.'.format(((dfh.nh_white>=0.9).sum()/len(dfh)*100)))
        print(dfh[['asian','hispanic', 'nh_black', 'nh_white']].mean())

        # trends over time
        female1, female2 = df[df.year>=2000].pc_female.mean(), df[df.year<2000].pc_female.mean()
        print('pc female increased from {:.0f}% (pre=2000) to {:.0f}% (2000-)'.format(female2*100, female1*100))

        nhw1, nhw2 = df[df.year>=2000].nh_white.mean(), df[df.year<2000].nh_white.mean()
        print('pc female increased from {:.0f}% (pre=2000) to {:.0f}% (2000-)'.format(nhw2*100, nhw1*100))

        # by course
        dfh = df[df.N_authors>0]
        female = dfh.groupby('courseid').pc_female.mean().sort_values(ascending=False)        
        print('{} courses with >50% female'.format((female>0.5).sum()))
        print('{} courses with 0% female'.format((female==0).sum()))
        nonHispWhite = dfh.groupby('courseid').nh_white.mean().sort_values()
        minNHW, maxNHW = nonHispWhite.min()*100, nonHispWhite.max()*100
        print('Range of NH white from {:.0f}% to {:.0f}%'.format(minNHW, maxNHW))

        # difference between R1/non R1 and landgrant/non-landgrant
        from scipy.stats import ttest_ind
        print('By R1/non R1')
        print(df.groupby('R1').pc_female.mean())
        print(ttest_ind(df.loc[(df.R1==1) & (pd.notnull(df.pc_female)), 'pc_female'], df.loc[(df.R1==0) & (pd.notnull(df.pc_female)), 'pc_female'], equal_var=False))
        print(df.groupby('R1').nh_white.mean())
        print(ttest_ind(df.loc[(df.R1==1) & (pd.notnull(df.nh_white)), 'nh_white'], df.loc[(df.R1==0) & (pd.notnull(df.nh_white)), 'nh_white'], equal_var=False))
        print('By Land Grant')
        print(df.groupby('landgrant').pc_female.mean())
        print(ttest_ind(df.loc[(df.landgrant==1) & (pd.notnull(df.pc_female)), 'pc_female'], df.loc[(df.landgrant==0) & (pd.notnull(df.pc_female)), 'pc_female'], equal_var=False))
        print(df.groupby('landgrant').nh_white.mean())
        print(ttest_ind(df.loc[(df.landgrant==1) & (pd.notnull(df.nh_white)), 'nh_white'], df.loc[(df.landgrant==0) & (pd.notnull(df.nh_white)), 'nh_white'], equal_var=False))
        
    def analyze_faculty(self):
        """Analyze race and gender distributions among planning faculty in North America
        List of faculty downloaded from http://www.scholarmetrics.com/about, July 6, 2020"""

        df = pd.read_excel(self.facultyFn)
        df['lastname']  = df.Name.apply(lambda x: x.split()[1].strip().title())
        df['firstname'] = df.Name.apply(lambda x: x.split()[0].strip().title())

        # add gender
        d = gender.Detector()
        df['gender'] = df.firstname.apply(lambda x: d.get_gender(x))
        df['female'] = df.gender.apply(lambda x: 1 if x in ['female','mostly_female'] else 0 if x in ['male','mostly_female'] else np.nan)
        # add race
        df = ethnicolr.pred_fl_reg_name(df,'lastname','firstname')

        outDf = df[['asian','hispanic', 'nh_black', 'nh_white','female']].mean() 
        outDf.to_csv(self.figpath+'summarytable_faculty.csv')

    def runall(self):
        self.loadData()
        self.plotData()
        self.reportStats()
        self.analyze_faculty()
        self.df.to_csv(self.outFn)

def test():
    """Tests syllabusAnalyzer using template1 and template 2
    Run in interactive mode."""
    path = ''
    correctResults = {'pc_female':0.583, 'asian':0.017, 'hispanic':0.023, 'nh_black':0.064, 'nh_white':0.895}
    N_authors = 11 
    for t in ['template1.csv', 'template2.csv']:
        sa = syllabusAnalyzer(path+t, None)
        sa.loadData()
        sa.addRaceGender()
        results = sa.outDf.mean()
        assert sa.outDf.N_authors.sum() == N_authors
        for col in correctResults:
            assert np.round(results[col],3) == correctResults[col] 
    print('Tests succeeded!')

if __name__ == '__main__':

    if len(sys.argv) not in [2,3]:
        print('Failed to run due to missing arguments')
        raise Exception('You need to specify the input file name, and (optionally) an output file name')
    
    inFn = sys.argv[1]
    outFn = sys.argv[2] if len(sys.argv)==3 else None
    syllabusAnalyzer(inFn, outFn).runall()
