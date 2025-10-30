"""
정혜원-예측변수 항목과 PISA 2018 변수 매핑
각 항목은 학생(STU), 학교(SCH), 교사(TCH) CSV 파일의 변수와 매핑됨
"""

variable_mapping = {
    # ========== 학생 관련 ==========
    "학생_배경": {
        "ESCS": {"변수": "ESCS", "출처": "STU", "설명": "Index of economic, social and cultural status"},
        "아버지의_사회경제직업수준": {"변수": "BFMJ2", "출처": "STU", "설명": "ISEI of father"},
        "어머니의_사회경제직업수준": {"변수": "BMMJ1", "출처": "STU", "설명": "ISEI of mother"},
        "아버지의_교육수준": {"변수": "FISCED", "출처": "STU", "설명": "Father's Education (ISCED)"},
        "어머니의_교육수준": {"변수": "MISCED", "출처": "STU", "설명": "Mother's Education (ISCED)"},
        "부모님의_교육수준": {"변수": "HISCED", "출처": "STU", "설명": "Highest Education of parents (ISCED)"},
        "가정의_문화적_보유자산": {"변수": "CULTPOSS", "출처": "STU", "설명": "Cultural possessions at home (WLE)"},
        "가정의_교육적_자산": {"변수": "HEDRES", "출처": "STU", "설명": "Home educational resources (WLE)"},
        "가정의_부유함": {"변수": "WEALTH", "출처": "STU", "설명": "Family wealth (WLE)"},
        "ICT_자산": {"변수": "ICTRES", "출처": "STU", "설명": "ICT resources (WLE)"},
        "이민_상황_지표": {
            "변수": ["ST019AQ01T", "ST019BQ01T", "ST019CQ01T", "ST021Q01TA", "IMMIG"],
            "출처": "STU", 
            "설명": "Immigration status: birth countries (student, mother, father), age of arrival, and immigration index"
        },
        "학력변경_횟수": {"변수": "CHANGE", "출처": "STU", "설명": "Number of changes in educational biography (Sum)"},
        "학년유예": {"변수": "REPEAT", "출처": "STU", "설명": "Grade Repetition"},
        "전학횟수": {"변수": "SCCHANGE", "출처": "STU", "설명": "Number of school changes"},
        "나이": {"변수": "AGE", "출처": "STU", "설명": "Age"},
        "가정에서_사용_가능한_ICT": {"변수": "ICTHOME", "출처": "STU", "설명": "ICT available at home"},
        "성별": {"변수": "ST004D01T", "출처": "STU", "설명": "Student (Standardized) Gender"},
        "집에서_주로_사용하는_언어": {"변수": "ST022Q01TA", "출처": "STU", "설명": "What language do you speak at home most of the time?"},
        "초등학교_재학_중_전학_여부": {"변수": "EC031Q01TA", "출처": "STU", "설명": "Did you change schools when you were attending <ISCED 1>?"},
        "중학교_재학_중_전학_여부": {"변수": "EC032Q01TA", "출처": "STU", "설명": "Did you change schools when you were attending <ISCED 2>?"},
        "학습_계열_변경_여부": {"변수": "EC033Q01NA", "출처": "STU", "설명": "Have you ever changed your <study programme>?"},
        "연간_가계_소득": {"변수": "PA042Q01TA", "출처": "STU", "설명": "What is your annual household income?"}
    },
    
    "학생_과정": {
        "읽기_전략_요약": {"변수": "METASUM", "출처": "STU", "설명": "Meta-cognition: summarising"},
        "읽기_전략_이해_및_기억": {"변수": "UNDREM", "출처": "STU", "설명": "Meta-cognition: understanding and remembering"},
        "디지털_읽기_전략_질과_신뢰성_평가": {"변수": "METASPAM", "출처": "STU", "설명": "Meta-cognition: assess credibility"},
        "읽기에_대한_즐거움": {"변수": "JOYREAD", "출처": "STU", "설명": "Joy/Like reading (WLE)"},
        "읽기에_대한_능력_지각": {"변수": "SCREADCOMP", "출처": "STU", "설명": "Self-concept of reading: Perception of competence (WLE)"},
        "읽기에_대한_어려움_지각": {"변수": "SCREADDIFF", "출처": "STU", "설명": "Self-concept of reading: Perception of difficulty (WLE)"},
        "긍정적_감정": {"변수": "SWBP", "출처": "STU", "설명": "Subjective well-being: Positive affect (WLE)"},
        "행복감": {"변수": "ST016Q01NA", "출처": "STU", "설명": "Overall, how satisfied are you with your life as a whole these days?"},
        "실패에_대한_두려움": {"변수": "GFOFAIL", "출처": "STU", "설명": "General fear of failure (WLE)"},
        "숙달_목표_지향": {"변수": "MASTGOAL", "출처": "STU", "설명": "Mastery goal orientation (WLE)"},
        "회복탄력성": {"변수": "RESILIENCE", "출처": "STU", "설명": "Resilience (WLE)"},
        "작업_완성도": {
            "변수": ["ST182Q03HA", "ST182Q04HA", "ST182Q05HA", "ST182Q06HA", "WORKMAST"],
            "출처": "STU",
            "설명": "Work mastery: 4 items + composite index (WLE)"
        },
        "총_공부시간": {"변수": "TMINS", "출처": "STU", "설명": "Learning time (minutes per week) - in total"},
        "국어_공부시간": {"변수": "LMINS", "출처": "STU", "설명": "Learning time (minutes per week) - <test language>"},
        "수학_공부시간": {"변수": "MMINS", "출처": "STU", "설명": "Learning time (minutes per week) - <Mathematics>"},
        "과학_공부시간": {"변수": "SMINS", "출처": "STU", "설명": "Learning time (minutes per week) - <science>"},
        "PISA_검사_노력": {"변수": "EFFORT1", "출처": "STU", "설명": "How much effort did you put into this test?"},
        "PISA_응답_노력": {"변수": "EFFORT2", "출처": "STU", "설명": "How much effort would you have invested?"},
        "PISA_검사_어려움_인식": {"변수": "PISADIFF", "출처": "STU", "설명": "Perception of difficulty of the PISA test (WLE)"},
        "삶의_만족도": {"변수": "ST016Q01NA", "출처": "STU", "설명": "Overall, how satisfied are you with your life as a whole these days?"},
        "따돌림_경험": {
            "변수": ["ST038Q03NA", "ST038Q04NA", "ST038Q05NA", "ST038Q06NA", "ST038Q07NA", "ST038Q08NA"],
            "출처": "STU",
            "설명": "During the past 12 months, how often: bullying experiences"
        },
        "무단결석_결과_지각": {
            "변수": ["ST062Q01TA", "ST062Q02TA", "ST062Q03TA"],
            "출처": "STU",
            "설명": "In the last two full weeks of school, how often: skipped/late"
        },
        "따돌림_의견": {
            "변수": ["ST207Q01HA", "ST207Q02HA", "ST207Q03HA", "ST207Q04HA", "ST207Q05HA"],
            "출처": "STU",
            "설명": "Agree: opinions about bullying"
        },
        "학습기회_텍스트_길이": {"변수": "ST154Q01HA", "출처": "STU", "설명": "<this academic year>, how many pages was the longest piece of text you had to read for your <test language lessons>?"},
        "다양한_자료_읽기_활동": {
            "변수": ["ST167Q01IA", "ST167Q02IA", "ST167Q03IA", "ST167Q04IA", "ST167Q05IA"],
            "출처": "STU",
            "설명": "How often do you read these materials because you want to?"
        },
        "책_읽는_방식": {"변수": "ST168Q01HA", "출처": "STU", "설명": "Which of the following statements best describes how you read books (on any topic)?"},
        "뉴스_읽는_방식": {"변수": "IC169Q01HA", "출처": "STU", "설명": "Which of the following statements best describes how you read the news"},
        "읽기_시간": {"변수": "ST175Q01IA", "출처": "STU", "설명": "About how much time do you usually spend reading for enjoyment?"},
        "온라인_읽기_활동": {
            "변수": ["ST176Q01IA", "ST176Q02IA", "ST176Q03IA", "ST176Q05IA", "ST176Q06IA", "ST176Q07IA"],
            "출처": "STU",
            "설명": "How often involved in: online reading activities"
        },
        "지능의_가변_이론": {"변수": "ST184Q01HA", "출처": "STU", "설명": "Agree: Your intelligence is something about you that you can't change very much."},
        "주관적_삶의_안녕감": {
            "변수": ["ST185Q01HA", "ST185Q02HA", "ST185Q03HA", "ST186Q05HA", "ST186Q06HA", "ST186Q07HA"],
            "출처": "STU",
            "설명": "Life satisfaction and emotional well-being"
        },
        "기술습득_여부": {
            "변수": ["EC151Q01WA", "EC151Q01WB", "EC151Q01WC", "EC151Q02WA", "EC151Q02WB", "EC151Q02WC",
                    "EC151Q03WA", "EC151Q03WB", "EC151Q03WC", "EC151Q04WA", "EC151Q04WB", "EC151Q04WC",
                    "EC151Q05WA", "EC151Q05WB", "EC151Q05WC"],
            "출처": "STU",
            "설명": "Acquired skills: career-related skills"
        },
        "정규수업외_추가교육": {
            "변수": ["EC154Q01IA", "EC154Q02IA", "EC154Q03IA", "EC154Q04HA", "EC154Q05IA",
                    "EC154Q06IA", "EC154Q07IA", "EC154Q08HA", "EC154Q09IA"],
            "출처": "STU",
            "설명": "Do you currently attend additional instruction?"
        },
        "등교전_공부시간": {"변수": ["EC158Q01HA", "EC158Q02HA"], "출처": "STU", "설명": "How long did you study in the morning before going to school?"},
        "방과후_공부시간": {"변수": ["EC159Q01HA", "EC159Q02HA"], "출처": "STU", "설명": "How long did you study after leaving school?"},
        "등교전_방과후_공부_이유": {
            "변수": ["EC163Q01HA", "EC163Q02HA", "EC163Q03HA", "EC163Q04HA", "EC163Q05HA", "EC163Q06HA", "EC163Q07HA"],
            "출처": "STU",
            "설명": "Why did you study before or after school?"
        },
        "해외_교환학생_프로그램_참여_희망": {"변수": "EC160Q01HA", "출처": "STU", "설명": "If you had the opportunity to participate in a student exchange programme [...], would you like to take part?"}
    },
    
    "진로_관련": {
        "학생이_기대하는_직업_지위": {"변수": "BSMJ", "출처": "STU", "설명": "Student's expected occupational status (SEI)"},
        "직업에_대한_정보": {"변수": "INFOCAR", "출처": "STU", "설명": "Information about careers (WLE)"},
        "학교_노동시장_정보": {"변수": "INFOJOB1", "출처": "STU", "설명": "Information about the labour market provided by the school (WLE)"},
        "학교외_노동시장_정보": {"변수": "INFOJOB2", "출처": "STU", "설명": "Information about the labour market provided outside of school (WLE)"},
        "최종_학력_기대": {
            "변수": ["ST225Q01HA", "ST225Q02HA", "ST225Q03HA", "ST225Q04HA", "ST225Q05HA", "ST225Q06HA"],
            "출처": "STU",
            "설명": "Do you expect to complete?"
        },
        "미래_직업_결정_중요_요소": {
            "변수": ["EC153Q01HA", "EC153Q02HA", "EC153Q03HA", "EC153Q04HA", "EC153Q05HA",
                    "EC153Q06HA", "EC153Q07HA", "EC153Q08HA", "EC153Q09HA", "EC153Q10HA", "EC153Q11HA"],
            "출처": "STU",
            "설명": "Importance for decisions about future occupation"
        }
    },
    
    "ICT": {
        "ICT에_대한_흥미": {"변수": "INTICT", "출처": "STU", "설명": "Interest in ICT (WLE)"},
        "ICT역량_인지": {"변수": "COMPICT", "출처": "STU", "설명": "Perceived ICT competence (WLE)"},
        "ICT_사용_자율성": {"변수": "AUTICT", "출처": "STU", "설명": "Perceived autonomy related to ICT use (WLE)"},
        "학교_사용가능_ICT": {"변수": "ICTSCH", "출처": "STU", "설명": "ICT available at school"},
        "학교_일상적_ICT_사용": {"변수": "USESCH", "출처": "STU", "설명": "Use of ICT at school in general (WLE)"},
        "학교외_ICT_사용_학교활동": {"변수": "HOMESCH", "출처": "STU", "설명": "Use of ICT outside of school (for school work activities) (WLE)"},
        "학교외_ICT_사용_레저": {"변수": "ENTUSE", "출처": "STU", "설명": "ICT use outside of school (leisure) (WLE)"},
        "사회적_상호작용_ICT": {"변수": "SOIAICT", "출처": "STU", "설명": "ICT as a topic in social interaction (WLE)"},
        "수업_ICT_사용": {"변수": "ICTCLASS", "출처": "STU", "설명": "Subject-related ICT use during lessons (WLE)"},
        "수업외_ICT_사용": {"변수": "ICTOUTSIDE", "출처": "STU", "설명": "Subject-related ICT use outside of lessons (WLE)"},
        "디지털기기_처음_사용_시기": {"변수": "IC002Q01HA", "출처": "STU", "설명": "How old were you when you first used a digital device?"},
        "인터넷_처음_접속_시기": {"변수": "IC004Q01HA", "출처": "STU", "설명": "How old were you when you first accessed the Internet?"},
        "학교_인터넷_사용_정도": {"변수": "IC005Q01TA", "출처": "STU", "설명": "During a typical weekday, for how long do you use the Internet at school?"},
        "학교외_인터넷_사용_정도": {
            "변수": ["IC006Q01TA", "IC007Q01TA"],
            "출처": "STU",
            "설명": "Internet use outside of school - weekday and weekend"
        },
        "수업_디지털기기_사용": {
            "변수": ["IC152Q01HA", "IC152Q02HA", "IC152Q03HA", "IC152Q04HA", "IC152Q05HA"],
            "출처": "STU",
            "설명": "Digital device used for learning or teaching during lessons within the last month"
        },
        "학교외_학교관련_디지털기기_사용": {"변수": "HOMESCH", "출처": "STU", "설명": "Use of ICT outside of school (for school work activities) (WLE)"}
    },
    
    "사회환경적": {
        "학교_소속감": {"변수": "BELONG", "출처": "STU", "설명": "Subjective well-being: Sense of belonging to school (WLE)"},
        "경쟁력": {"변수": "COMPETE", "출처": "STU", "설명": "Competitiveness (WLE)"},
        "학교_경쟁_인식": {"변수": "PERCOMP", "출처": "STU", "설명": "Perception of competitiveness at school (WLE)"},
        "학생_협동": {"변수": "PERCOOP", "출처": "STU", "설명": "Perception of cooperation at school (WLE)"},
        "학습활동_태도": {"변수": "ATTLNACT", "출처": "STU", "설명": "Attitude towards school: learning activities (WLE)"},
        "교사_흥미_인식": {"변수": "TEACHINT", "출처": "STU", "설명": "Perceived teacher's interest (WLE)"},
        "교사_직접적_지시": {"변수": "DIRINS", "출처": "STU", "설명": "Teacher-directed instruction (WLE)"},
        "수업_피드백": {"변수": "PERFEED", "출처": "STU", "설명": "Perceived feedback (WLE)"},
        "지시_순응": {"변수": "DISCLIMA", "출처": "STU", "설명": "Disciplinary climate in test language lessons (WLE)"},
        "읽기_참여_교사_지지": {"변수": "STIMREAD", "출처": "STU", "설명": "Teacher's stimulation of reading engagement perceived by student (WLE)"},
        "국어수업_징벌적_분위기": {"변수": "DISCLIMA", "출처": "STU", "설명": "Disciplinary climate in test language lessons (WLE)"},
        "국어수업_교사_지지": {"변수": "TEACHSUP", "출처": "STU", "설명": "Teacher support in test language lessons (WLE)"},
        "부모_정서적_지지": {"변수": "EMOSUPS", "출처": "STU", "설명": "Parents' emotional support perceived by student (WLE)"},
        "유아기_교육과_돌봄": {"변수": "DURECEC", "출처": "STU", "설명": "Duration in early childhood education and care"},
        "학교공부_주변_지원": {
            "변수": ["EC155Q01DA", "EC155Q02DA", "EC155Q03DA", "EC155Q04HA", "EC155Q05HA"],
            "출처": "STU",
            "설명": "How often do the following people work with you on your schoolwork?"
        },
        "교사_지원": {
            "변수": ["ST100Q01TA", "ST100Q02TA", "ST100Q03TA", "ST100Q04TA"],
            "출처": "STU",
            "설명": "Teacher support in test language lessons"
        },
        "학습기회_자료": {
            "변수": ["ST150Q01IA", "ST150Q02IA", "ST150Q03IA", "ST150Q04HA"],
            "출처": "STU",
            "설명": "During the last month, how often did you have to read for school"
        },
        "학습기회_국어수업_과제": {
            "변수": ["ST153Q01HA", "ST153Q02HA", "ST153Q03HA", "ST153Q04HA", "ST153Q05HA",
                    "ST153Q06HA", "ST153Q08HA", "ST153Q09HA", "ST153Q10HA"],
            "출처": "STU",
            "설명": "When you have to read, does the teacher ask you to"
        },
        "디지털_국어_학습기회": {
            "변수": ["ST158Q01HA", "ST158Q02HA", "ST158Q03HA", "ST158Q04HA", "ST158Q05HA", "ST158Q06HA", "ST158Q07HA"],
            "출처": "STU",
            "설명": "Taught at school: digital literacy skills"
        }
    },
    
    # ========== 부모 관련 ==========
    "부모_관련": {
        "현재_가정_학습_지원": {"변수": "CURSUPP", "출처": "STU", "설명": "Current parental support for learning at home (WLE)"},
        "과거_가정_학습_지원": {"변수": "PRESUPP", "출처": "STU", "설명": "Previous parental support for learning at home (WLE)"},
        "부모_정서적_지원": {"변수": "EMOSUPP", "출처": "STU", "설명": "Parents' emotional support (WLE)"},
        "부모_읽기_즐거움": {"변수": "JOYREADP", "출처": "STU", "설명": "Parents' enjoyment of reading (WLE)"},
        "학부모_참여_학교정책": {"변수": "PASCHPOL", "출처": "STU", "설명": "School policies for parental involvement (WLE)"},
        "부모_인식_학교_질": {"변수": "PQSCHOOL", "출처": "STU", "설명": "Parents' perceived school quality (WLE)"},
        "자녀_10살_읽기_빈도": {
            "변수": ["PA156Q01HA", "PA156Q02HA", "PA156Q03HA", "PA156Q04HA"],
            "출처": "STU",
            "설명": "Thinking back to when your child was about 10 years old, how often would he or she read"
        },
        "부모_취미_독서시간": {"변수": "PA159Q01HA", "출처": "STU", "설명": "About how much time do you usually spend reading for enjoyment?"},
        "부모_다양한_읽기_자료": {
            "변수": ["PA160Q01HA", "PA160Q02HA", "PA160Q03HA", "PA160Q04HA", "PA160Q05HA",
                    "PA161Q01HA", "PA161Q02HA", "PA161Q03HA", "PA161Q05HA", "PA161Q06HA", "PA161Q07HA"],
            "출처": "STU",
            "설명": "How often do you read these types of texts / reading activities"
        },
        "부모_책_읽는_방식": {"변수": "PA162Q01HA", "출처": "STU", "설명": "Which of the following statements best describes how you read books"},
        "부모_뉴스_읽는_방식": {"변수": "PA163Q01HA", "출처": "STU", "설명": "Which of the following statements best describes how you read the news"},
        "12개월_교육비": {"변수": "PA041Q01TA", "출처": "STU", "설명": "In the last twelve months, about how much would you have paid to educational providers for services?"},
        "자녀_학교_선택권": {
            "변수": ["PA005Q01TA", "PA006Q01TA"],
            "출처": "STU",
            "설명": "School choice availability and importance"
        },
        "자녀_학교_선택_중요_요소": {
            "변수": ["PA006Q02TA", "PA006Q03TA", "PA006Q04TA", "PA006Q05TA", "PA006Q06TA",
                    "PA006Q07TA", "PA006Q08TA", "PA006Q09TA", "PA006Q10TA", "PA006Q11TA",
                    "PA006Q12HA", "PA006Q13HA", "PA006Q14HA"],
            "출처": "STU",
            "설명": "Importance for choosing a school"
        },
        "학부모_학교활동_참여도": {
            "변수": ["PA008Q01TA", "PA008Q02TA", "PA008Q03TA", "PA008Q04TA", "PA008Q05TA",
                    "PA008Q06NA", "PA008Q07NA", "PA008Q08NA", "PA008Q09NA", "PA008Q10NA"],
            "출처": "STU",
            "설명": "<the last academic year>: parent participation in school activities"
        },
        "학부모_학교활동_불참_사유": {
            "변수": ["PA009Q01NA", "PA009Q02NA", "PA009Q03NA", "PA009Q04NA", "PA009Q05NA",
                    "PA009Q06NA", "PA009Q07NA", "PA009Q08NA", "PA009Q09NA", "PA009Q10NA", "PA009Q11NA"],
            "출처": "STU",
            "설명": "<the last academic year>, participation hindered"
        },
        "자녀_영유아_교육_참여_여부": {
            "변수": ["PA018Q01NA", "PA018Q02NA", "PA018Q03NA"],
            "출처": "STU",
            "설명": "Child regularly attended prior to <grade 1 in ISCED 1>"
        },
        "자녀_영유아_교육_참여_나이": {
            "변수": ["PA177Q01HA", "PA177Q02HA", "PA177Q03HA", "PA177Q04HA",
                    "PA177Q05HA", "PA177Q06HA", "PA177Q07HA", "PA177Q08HA"],
            "출처": "STU",
            "설명": "Ages child attended <early childhood education and care arrangement>"
        },
        "자녀_영유아_교육_참여_이유": {
            "변수": ["PA180Q01HA", "PA182Q01HA", "PA183Q01HA"],
            "출처": "STU",
            "설명": "Early childhood education participation reason and duration"
        },
        "자녀_영유아_교육_참여_정도": {"변수": "PA182Q01HA", "출처": "STU", "설명": "Hours per week child attended a <early childhood education and care arrangement>"},
        "자녀_초등1학년_활동_언어": {"변수": "PA155Q01IA", "출처": "STU", "설명": "In what language did most of the activities in the previous question take place?"},
        "사회이슈_역사_관심": {
            "변수": ["PA169Q01HA", "PA169Q02HA", "PA169Q03HA", "PA169Q04HA", "PA169Q05HA", "PA169Q06HA"],
            "출처": "STU",
            "설명": "How interested are you in the following issues?"
        },
        "자녀_기대_학력": {
            "변수": ["PA172Q01WA", "PA172Q02WA", "PA172Q03WA", "PA172Q04WA", "PA172Q05WA", "PA172Q06WA"],
            "출처": "STU",
            "설명": "Which of the following do you expect your child to complete?"
        },
        "자녀_초등_추가_국어수업": {
            "변수": ["PA175Q01HA", "PA175Q02HA"],
            "출처": "STU",
            "설명": "Did your child attend the following additional instructions during [ISCED 1]?"
        },
        "학부모_설문_응답자": {
            "변수": ["PA001Q01TA", "PA001Q02TA", "PA001Q03TA"],
            "출처": "STU",
            "설명": "Who will complete this questionnaire?"
        }
    },
    
    # ========== 교사 관련 ==========
    "교사_관련": {
        "교사만족도_근무환경": {"변수": "SATJOB", "출처": "TCH", "설명": "Teacher's satisfaction with the current job environment (WLE)"},
        "교사만족도_직": {"변수": "SATTEACH", "출처": "TCH", "설명": "Teacher's satisfaction with teaching profession (WLE)"},
        "교사효능감_학급운영": {"변수": "SEFFCM", "출처": "TCH", "설명": "Teacher's self-efficacy in classroom management (WLE)"},
        "교사효능감_교수환경": {"변수": "SEFFINS", "출처": "TCH", "설명": "Teacher's self-efficacy in instructional settings (WLE)"},
        "교사효능감_긍정적_관계": {"변수": "SEFFREL", "출처": "TCH", "설명": "Teacher's self-efficacy in maintaining positive relations with students (WLE)"},
        "fulltime_여부": {"변수": "EMPLTIM", "출처": "TCH", "설명": "Teacher employment time - dichotomous"},
        "교사_교환_조정": {"변수": "EXCHT", "출처": "TCH", "설명": "Exchange and co-ordination for teaching (WLE)"},
        "교사_피드백": {"변수": "FEEDBACK", "출처": "TCH", "설명": "Feedback provided by the teachers (WLE)"},
        "교사_엄격한_정의": {"변수": "OTT1", "출처": "TCH", "설명": "Originally trained teacher (strict definition): standard teacher training"},
        "교사_광범위한_정의": {"변수": "OTT2", "출처": "TCH", "설명": "Originally trained teacher (wide definition): standard, in-service, or work-based teacher training"},
        "교사_교육자료_부족": {"변수": "TCEDUSHORT", "출처": "TCH", "설명": "Teacher's view on educational material shortage (WLE)"},
        "교사_교직원_부족": {"변수": "TCSTAFFSHORT", "출처": "TCH", "설명": "Teacher's view on staff shortage (WLE)"},
        "교사_ICT_응용": {"변수": "TCICTUSE", "출처": "TCH", "설명": "Teacher's use of specific ICT applications (WLE)"},
        "독해력_기회": {"변수": "TCOTLCOMP", "출처": "TCH", "설명": "Opportunity to learn (OTL) aspects of reading comprehension (WLE)"},
        "교사_전문지식_효능감": {
            "변수": ["TC152Q01HA", "TC152Q02HA", "TC152Q03HA", "TC152Q04HA"],
            "출처": "TCH",
            "설명": "Teacher's professional knowledge and efficacy"
        },
        "교사_신념_효능감": {
            "변수": ["TC198Q01HA", "TC198Q02HA", "TC198Q03HA", "TC198Q04HA",
                    "TC198Q05HA", "TC198Q06HA", "TC198Q07HA", "TC198Q08HA"],
            "출처": "TCH",
            "설명": "Teacher's beliefs and efficacy about teaching"
        },
        "교직_견해": {
            "변수": ["TC198Q08HA", "TC198Q09HA", "TC198Q10HA"],
            "출처": "TCH",
            "설명": "Views on teaching profession"
        },
        "12개월_전문성_계발": {
            "변수": ["TC193Q01HA", "TC193Q02HA", "TC193Q03HA", "TC193Q04HA", "TC193Q05HA"],
            "출처": "TCH",
            "설명": "During the last 12 months, participated in professional development"
        },
        "전문성_계발_참여": {
            "변수": ["TC020Q01NA", "TC020Q02NA", "TC020Q03NA", "TC020Q04NA", "TC020Q05NA", "TC020Q06NA"],
            "출처": "TCH",
            "설명": "During the last 12 months, participated in various professional development activities"
        },
        "전문성_계발_의무": {"변수": "TC021Q01NA", "출처": "TCH", "설명": "Are you required to take part in professional development activities?"},
        "전문성_계발_주제": {
            "변수": ["TC045Q01NA", "TC045Q01NB", "TC045Q02NA", "TC045Q02NB", "TC045Q03NA", "TC045Q03NB",
                    "TC045Q04NA", "TC045Q04NB", "TC045Q05NA", "TC045Q05NB", "TC045Q06NA", "TC045Q06NB",
                    "TC045Q07NA", "TC045Q07NB", "TC045Q08NA", "TC045Q08NB", "TC045Q09NA", "TC045Q09NB",
                    "TC045Q10NA", "TC045Q10NB", "TC045Q11NA", "TC045Q11NB", "TC045Q12NA", "TC045Q12NB",
                    "TC045Q13NA", "TC045Q13NB", "TC045Q14NA", "TC045Q14NB", "TC045Q15NA", "TC045Q15NB",
                    "TC045Q16HA", "TC045Q16HB", "TC045Q17HA", "TC045Q17HB", "TC045Q18HA", "TC045Q18HB"],
            "출처": "TCH",
            "설명": "Included in teacher education, training or other qualification and professional development"
        },
        "학습평가_방법": {
            "변수": ["TC054Q01NA", "TC054Q02NA", "TC054Q03NA", "TC054Q04NA", "TC054Q05NA", "TC054Q06NA", "TC054Q07NA"],
            "출처": "TCH",
            "설명": "Assessing student learning, how often"
        },
        "읽기_수업_여부": {
            "변수": ["TC018Q01NA", "TC018Q01NB"],
            "출처": "TCH",
            "설명": "Reading-related instruction"
        },
        "가장_긴_지문": {"변수": "TC164Q01HA", "출처": "TCH", "설명": "<this academic year>, how many pages was the longest piece of text your students had to read"},
        "디지털_읽기_역량_수업": {
            "변수": ["TC166Q01HA", "TC166Q02HA", "TC166Q03HA", "TC166Q04HA", "TC166Q05HA", "TC166Q06HA", "TC166Q07HA"],
            "출처": "TCH",
            "설명": "In your lessons, have you ever taught: digital literacy skills"
        },
        "교사_디지털_읽기_활동": {
            "변수": ["TC155Q02HA", "TC155Q03HA", "TC155Q04HA", "TC155Q05HA", "TC155Q06HA", "TC155Q07HA"],
            "출처": "TCH",
            "설명": "In your lessons, how often do you teach: reading strategies"
        },
        "학교_디지털기기_규정": {"변수": "TC184Q01HA", "출처": "TCH", "설명": "Does your school have a policy concerning the use of digital devices for teaching?"},
        "교사_책_읽는_방식": {"변수": "TC172Q01HA", "출처": "TCH", "설명": "Best describes how you read books"},
        "교사_뉴스_읽는_방식": {"변수": "TC173Q01HA", "출처": "TCH", "설명": "Best describes how you read the news"},
        "수업외_업무_독서시간": {"변수": "TC175Q01HA", "출처": "TCH", "설명": "About how much time per week do you spend reading for your work out of your classes?"},
        "출생_국가": {"변수": "TC186Q01HA", "출처": "TCH", "설명": "Country of birth"},
        "다른_국가_공부": {"변수": "TC188Q01HA", "출처": "TCH", "설명": "Studied in a country other than [country of test]"},
        "교사_근무_연수": {
            "변수": ["TC007Q01NA", "TC007Q02NA"],
            "출처": "TCH",
            "설명": "How many years of work experience do you have?"
        },
        "나이": {"변수": "TC002Q01NA", "출처": "TCH", "설명": "How old are you?"},
        "교사_자격증_취득": {
            "변수": ["TC014Q01HA", "TC015Q01NA", "TC018Q01NA", "TC018Q01NB"],
            "출처": "TCH",
            "설명": "Teacher qualification and training"
        }
    },
    
    # ========== 학교 관련 ==========
    "학교_관련": {
        "학교크기": {"변수": "SCHSIZE", "출처": "SCH", "설명": "School Size (Sum)"},
        "학생_교사_비율": {"변수": "STRATIO", "출처": "SCH", "설명": "Student-Teacher ratio"},
        "설립유형": {
            "변수": ["SC013Q01TA", "SCHLTYPE"],
            "출처": "SCH",
            "설명": "Is your school a public or a private school? / School Ownership"
        },
        "교사_전체_수": {"변수": "TOTAT", "출처": "SCH", "설명": "Total number of all teachers at school"},
        "정규직_교사_비율": {"변수": "PROATCE", "출처": "SCH", "설명": "Index proportion of all teachers fully certified"},
        "학사졸업_교사_비율": {"변수": "PROAT5AB", "출처": "SCH", "설명": "Index proportion of all teachers ISCED LEVEL 5A Bachelor"},
        "석사졸업_교사_비율": {"변수": "PROAT5AM", "출처": "SCH", "설명": "Index proportion of all teachers ISCED LEVEL 5A Master"},
        "박사졸업_교사_비율": {"변수": "PROAT6", "출처": "SCH", "설명": "Index proportion of all teachers ISCED LEVEL 6"},
        "학생_1인당_컴퓨터_수": {"변수": "RATCMP1", "출처": "SCH", "설명": "Number of available computers per student at modal grade"},
        "인터넷_연결_컴퓨터_비율": {"변수": "RATCMP2", "출처": "SCH", "설명": "Proportion of available computers that are connected to the Internet"},
        "학습도구_부족": {"변수": "EDUSHORT", "출처": "SCH", "설명": "Shortage of educational material (WLE)"},
        "학습보조원_부족": {"변수": "STAFFSHORT", "출처": "SCH", "설명": "Shortage of educational staff (WLE)"},
        "교과외_활동_제공": {"변수": "CREACTIV", "출처": "SCH", "설명": "Creative extra-curricular activities (Sum)"},
        "학생_행동_학습방해": {"변수": "STUBEHA", "출처": "SCH", "설명": "Student behaviour hindering learning (WLE)"},
        "교사_행동_학습방해": {"변수": "TEACHBEHA", "출처": "SCH", "설명": "Teacher behaviour hindering learning (WLE)"},
        "학교위치": {"변수": "SC001Q01TA", "출처": "SCH", "설명": "Which of the following definitions best describes the community in which your school is located?"},
        "학교_선택": {"변수": "SC011Q01TA", "출처": "SCH", "설명": "Which of the following statements best describes the schooling available to students in your location?"},
        "교사_수": {
            "변수": ["SC018Q01TA01", "SC018Q01TA02"],
            "출처": "SCH",
            "설명": "Teachers in TOTAL: Full-time and Part-time"
        },
        "학생_중퇴_비율": {"변수": "SC164Q01HA", "출처": "SCH", "설명": "In the last full academic year, what proportion of students in final grade left school without a certificate?"},
        "입학_사정_고려사항": {
            "변수": ["SC012Q01TA", "SC012Q02TA", "SC012Q03TA", "SC012Q04TA", "SC012Q05TA", "SC012Q06TA", "SC012Q07TA"],
            "출처": "SCH",
            "설명": "Student admission to school criteria"
        },
        "전문성_계발_연수_참여_비율": {"변수": "SC025Q01NA", "출처": "SCH", "설명": "During the last three months, what percentage of teaching staff attended a programme of professional development?"},
        "성취도_책무성_활용": {
            "변수": ["SC036Q01TA", "SC036Q02TA", "SC036Q03NA"],
            "출처": "SCH",
            "설명": "Use of achievement data in school"
        },
        "질_점검_개선_대책": {
            "변수": ["SC037Q01TA", "SC037Q02TA", "SC037Q03TA", "SC037Q04TA", "SC037Q05NA",
                    "SC037Q06NA", "SC037Q07TA", "SC037Q08TA", "SC037Q09TA", "SC037Q10NA"],
            "출처": "SCH",
            "설명": "Quality assurance at school"
        },
        "1학년_능력별_편성": {
            "변수": ["SC042Q01TA", "SC042Q02TA"],
            "출처": "SCH",
            "설명": "School's policy for <national modal grade for 15-year-olds>: ability grouping"
        },
        "1학년_학생_특성": {
            "변수": ["SC048Q01NA", "SC048Q02NA", "SC048Q03NA"],
            "출처": "SCH",
            "설명": "Percentage <national modal grade for 15-year-olds>: student characteristics"
        },
        "학생_학습_지원": {
            "변수": ["SC052Q01NA", "SC052Q02NA", "SC052Q03HA"],
            "출처": "SCH",
            "설명": "For 15-year old students, school provides study help"
        },
        "학부모_참여_비율": {
            "변수": ["SC064Q01TA", "SC064Q02TA", "SC064Q03TA", "SC064Q04NA"],
            "출처": "SCH",
            "설명": "Proportion of parents participating in school activities"
        },
        "모국어_아닌_학생_평등정책": {
            "변수": ["SC150Q01IA", "SC150Q02IA", "SC150Q03IA", "SC150Q04IA", "SC150Q05IA"],
            "출처": "SCH",
            "설명": "School's equity-oriented policies"
        },
        "정규수업외_추가_국어수업": {"변수": "SC152Q01HA", "출처": "SCH", "설명": "Does your school offer additional <test language> lessons during the usual school hours?"},
        "국어수업_추가_목적": {"변수": "SC160Q01WA", "출처": "SCH", "설명": "What is the purpose of these additional <test language> lessons?"},
        "학생_평가결과_사용": {
            "변수": ["SC154Q01HA", "SC154Q02WA", "SC154Q03WA", "SC154Q04WA", "SC154Q05WA",
                    "SC154Q06WA", "SC154Q07WA", "SC154Q08WA", "SC154Q09HA", "SC154Q10WA", "SC154Q11HA"],
            "출처": "SCH",
            "설명": "School's use of assessments of students"
        },
        "ICT_도구_준비_수준": {
            "변수": ["SC155Q01HA", "SC155Q02HA", "SC155Q03HA", "SC155Q04HA", "SC155Q05HA",
                    "SC155Q06HA", "SC155Q07HA", "SC155Q08HA", "SC155Q09HA", "SC155Q10HA", "SC155Q11HA"],
            "출처": "SCH",
            "설명": "School's capacity using digital devices"
        },
        "ICT_활용_전략": {
            "변수": ["SC156Q01HA", "SC156Q02HA", "SC156Q03HA", "SC156Q04HA", "SC156Q05HA",
                    "SC156Q06HA", "SC156Q07HA", "SC156Q08HA"],
            "출처": "SCH",
            "설명": "At school: ICT-related policies and programs"
        },
        "해외학교_교사_교환": {"변수": "SC159Q01HA", "출처": "SCH", "설명": "Does your school host visiting teachers from other countries?"},
        "진로지도_담당자": {
            "변수": ["SC161Q01SA", "SC161Q02SA", "SC161Q03SA", "SC161Q04SA", "SC161Q05SA"],
            "출처": "SCH",
            "설명": "Main responsibility for career guidance at school"
        },
        "진로지도_방법": {"변수": "SC162Q01SA", "출처": "SCH", "설명": "If career guidance is available at your school, which of the statements below best describes the situation"}
    }
}

# 추가 정보: 변수 출처별 카운트
def count_variables_by_source():
    """변수 출처별 개수 카운트"""
    counts = {"STU": 0, "SCH": 0, "TCH": 0}
    
    for category, items in variable_mapping.items():
        if category not in ["학생_배경", "학생_과정", "진로_관련", "ICT", "사회환경적", "부모_관련", "교사_관련", "학교_관련"]:
            continue
            
        for item_name, item_info in items.items():
            source = item_info.get("출처", "")
            if source in counts:
                counts[source] += 1
    
    return counts

if __name__ == "__main__":
    print("=" * 80)
    print("정혜원-예측변수 → PISA 2018 변수 매핑")
    print("=" * 80)
    
    # 카테고리별 매핑 출력
    for category, items in variable_mapping.items():
        if category in ["학생_배경", "학생_과정", "진로_관련", "ICT", "사회환경적", "부모_관련", "교사_관련", "학교_관련"]:
            print(f"\n[{category}] - {len(items)}개 항목")
            print("-" * 80)
            
            for item_name, item_info in items.items():
                var = item_info.get("변수", "")
                source = item_info.get("출처", "")
                desc = item_info.get("설명", "")
                
                if isinstance(var, list):
                    print(f"  • {item_name}: {len(var)}개 변수 ({source})")
                    print(f"    - {', '.join(var[:3])}{'...' if len(var) > 3 else ''}")
                else:
                    print(f"  • {item_name}: {var} ({source})")
                print(f"    {desc[:100]}...")
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("변수 출처별 통계")
    print("=" * 80)
    counts = count_variables_by_source()
    for source, count in counts.items():
        print(f"{source}: {count}개 항목")
