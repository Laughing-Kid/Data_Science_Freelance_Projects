<!--
COPYRIGHT NOTICE
Copyright (c) 2009  Scott C. Neu and Arthur W. Toga

See README license for license notices.
-->












<HTML>












<HEAD>
<!--
COPYRIGHT NOTICE
Copyright (c) 2021  Scott C. Neu and Arthur W. Toga

See README.license for license notices.
-->



















<html>
<head>


















































 
 
 









































<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
<meta name="description"
  content="The Image Data Archive at the Laboratory of Neuro Imaging (IDA) provides a safe repository for medical imaging data.  LONI seeks to improve the understanding of the brain in health and disease through the development of algorithms and approaches for the comprehensive and quantitative mapping of its structure and function." />
<meta name="keywords"
  content="LONI, IDA, Image Data Archive, Laboratory of Neuro Imaging, medical image data, DICOM, MINC, Analyze, NIfTI, neuroimaging analyses, brain atlases" />
<meta name="author" content="Crawford" />
<meta name="copyright"
  content="Copyright 1997-2022 Laboratory of Neuro Imaging, All Rights Reserved" />
<meta name="robots" content="all" />
<meta name="revisit-after" content="5 days" />
<meta name="rating" content="safe for kids" />


<script type="text/javascript" src="https://ida.loni.usc.edu/services/jquery/jquery-1.12.0.min.js"></script>
<script type="text/javascript" src="https://ida.loni.usc.edu/services/jquery/jquery-ui-1.9.2.min.js"></script>
<script type="text/javascript" src="https://ida.loni.usc.edu/services/jquery/jquery-migrate-1.2.1.min.js"></script>


<link rel="stylesheet" type="text/css" href="https://ida.loni.usc.edu/services/CSS/ida.css" />
<link rel="stylesheet" type="text/css" href="https://ida.loni.usc.edu/services/CSS/home/header.css" />
<link rel="stylesheet" type="text/css" href="https://ida.loni.usc.edu/services/CSS/home/footer.css" />
<link rel="stylesheet" type="text/css" href="https://ida.loni.usc.edu/services/CSS/home/component.css" />
<link type="text/css" href="https://ida.loni.usc.edu/services/jquery/smoothness/jquery-ui-1.8.14.custom.css"
  rel="stylesheet" />






<!-- Use Google Analytics on production servers -->

  <script type="text/javascript">
            var _gaq = _gaq || [];
            _gaq.push(['_setAccount', 'UA-11758345-3']);
            _gaq.push(['_setDomainName', 'loni.usc.edu']);
            _gaq.push(['_trackPageview']);

            (function() {
                var ga = document.createElement('script');
                ga.type = 'text/javascript';
                ga.async = true;
                ga.src = ('https:' == document.location.protocol ? 'https://ssl'
                : 'http://www') +
                         '.google-analytics.com/ga.js';
                var s = document.getElementsByTagName('script')[0];
                s.parentNode.insertBefore(ga, s);
            })();
        </script>


<style type="text/css">
.project-bg {
    background: url("https://ida.loni.usc.edu/services/Images/home/bg.jpg") repeat;
}
</style>
<script type="text/javascript">
     if($("body").hasClass("login")) {
         $("body").css({
             backgroundImage:"url('https://ida.loni.usc.edu/services/Images/home/bg.jpg')",
             backgroundRepeat:"repeat"
         }); 
     } 

     $(document).ready(function() {
         $("form[name='form1'] input").keypress(function(e) {
             if(e.charCode == 13 || e.keyCode == 13) {
                   if($("form[name='form1'] #userEmail").val() && 
                      $("form[name='form1'] #userPassword").val()) {
                       $("form[name='form1']").submit();
                   }
             }
         });
     });
</script>

</head>

<body>
  
  
  

  <!-- Header starts -->
  <div class="header project-bg">

    <!-- USC logo and ida depending on status -->
    <div class="header-top-bar">
      
      <div class="header-top-bar-inner">
        
          
          
          
          
        
        <a class="header-usc-logo" href="https://www.usc.edu" target="_blank"></a>
      </div>
    </div>

    
    
    
    
    
    
    

      
      
      

      
      
      

      
      
      
      
      
      
        
        

          <div class="header-inner">
            <!-- project or ida logo -->
            
            
            <div class="header-logo-container">
              
                
                  <a href="https://ida.loni.usc.edu/login.jsp"><img src="https://ida.loni.usc.edu/services/Images/home/ida_logo.png" /></a>
                
                
              
            </div>

            <!-- space between logo and login box-->
            <div class="header-inner-filler-div"></div>



            
            <div class="header-login-container">
              
              
                
                
                
                  
                    
                    

                      <div class="header-public-menu">
                        <div class="header-public-menu-button">
                           <a href="https://ida.loni.usc.edu/login.jsp?project=">Home</a>
                        </div>
                        <div class="header-public-menu-button">
                           <a href="https://ida.loni.usc.edu/services/Menu/About.jsp?project=">About</a>
                        </div>
                        <div class="header-public-menu-button">
                          <a id="headerSupportAnchor" href="https://ida.loni.usc.edu/services/Menu/Support.jsp?project=" target="_blank">Support</a>
                        </div>
                      </div>

                      <!-- login -->
                      <div class="header-user-login-box">
                        <form name="form1" method="post" action="https://ida.loni.usc.edu/login.jsp?project=&page=HOME" class="header-no-pad">

                          <!-- username -->
                          <input type="text" name="userEmail" size="15" maxlength="60"
                            id="userEmail" class="header-public-menu-email"
                            placeholder=" Email" />

                          <!-- password -->
                          <input type="password" name="userPassword" size="15"
                            maxlength="35" id="userPassword"
                            class="header-public-menu-password" placeholder=" Password" />

                          <!-- login button -->
                          <div class="header-login-button">
                            <div class="header-login-button-inside"
                                 onclick="javascript:document.form1.submit();">LOGIN</div>
                          </div>

                          
                           
                             
                             
                            	<div class="header-error">
                               		
                                </div>
                                
                                 
		                         <div class="header-login-options">
		                          <div id="header-signup" class="header-sign-up-button">
		                            <nobr>Instant free signup</nobr>
		                          </div>
		
		                          <div class="header-login-options-filler-div"></div>
		                          <div class="header-forgot-password-button">
		                            <a href="https://ida.loni.usc.edu/services/RecoverPassword.jsp">Forgot Password?</a>
		                          </div>
		                        </div>
		
		                        <div id="header-signup-ie" class="header-forgot-password-button">
		                          <a href="https://ida.loni.usc.edu/services/NewUser.jsp">Instant free signup</a>
		                        </div>
                             
                           
                        </form>
                      </div>

                    

                    
                    
                  
                
              
            </div>              
          </div>

          
          
          
          
          
            
          
        
      
      
          
          <script src="https://ida.loni.usc.edu/services/js/cbpHorizontalMenu.js"></script>
          <script>
            $(document).ready(function() {
                cbpHorizontalMenu.init();
            });
            
            // Navigate to Signup page
            $('#header-signup').click(function() {
                window.location = "https://ida.loni.usc.edu/services/NewUser.jsp";
            });

            // Logout function
            $('#signOut').click(function() {
                window.location = "https://ida.loni.usc.edu/login.jsp?project=&page=HOME&logOut=true";
            });
            
            if ($.browser.msie && $.browser.version == '8.0') {
                $('#header-signup').hide();
                $('#header-signup-ie').show();
            }
            else {
                $('#header-signup').show();
                $('#header-signup-ie').hide();
            }
          </script>
    
  </div>
  <!-- header ends -->
</body>
</html>




<script language="JavaScript" type="text/JavaScript">
<!--
if(window.attachEvent)
    window.attachEvent("onload",setListeners);

function setListeners(){
    inputList = document.getElementsByTagName("INPUT");
    for(i=0;i<inputList.length;i++){
      inputList[i].attachEvent("onpropertychange",restoreStyles);
      inputList[i].style.backgroundColor = "";
    }
    selectList = document.getElementsByTagName("SELECT");
    for(i=0;i<selectList.length;i++){
      selectList[i].attachEvent("onpropertychange",restoreStyles);
      selectList[i].style.backgroundColor = "";
    }
  }

function restoreStyles(){
    if(event.srcElement.style.backgroundColor != "")
      event.srcElement.style.backgroundColor = "";
  }

function MM_preloadImages() { //v3.0
  var d=document; if(d.images){ if(!d.MM_p) d.MM_p=new Array();
    var i,j=d.MM_p.length,a=MM_preloadImages.arguments; for(i=0; i<a.length; i++)
    if (a[i].indexOf("#")!=0){ d.MM_p[j]=new Image; d.MM_p[j++].src=a[i];}}
}

function MM_findObj(n, d) { //v4.01
  var p,i,x;  if(!d) d=document; if((p=n.indexOf("?"))>0&&parent.frames.length) {
    d=parent.frames[n.substring(p+1)].document; n=n.substring(0,p);}
  if(!(x=d[n])&&d.all) x=d.all[n]; for (i=0;!x&&i<d.forms.length;i++) x=d.forms[i][n];
  for(i=0;!x&&d.layers&&i<d.layers.length;i++) x=MM_findObj(n,d.layers[i].document);
  if(!x && d.getElementById) x=d.getElementById(n); return x;
}

function MM_swapImage() { //v3.0
  var i,j=0,x,a=MM_swapImage.arguments; document.MM_sr=new Array; for(i=0;i<(a.length-2);i+=3)
   if ((x=MM_findObj(a[i]))!=null){document.MM_sr[j++]=x; if(!x.oSrc) x.oSrc=x.src; x.src=a[i+2];}
}

function MM_goToURL() { //v3.0
  var i, args=MM_goToURL.arguments; document.MM_returnValue = false;
  for (i=0; i<(args.length-1); i+=2) eval(args[i]+".location='"+args[i+1]+"'");
}

function MM_swapImgRestore() { //v3.0
  var i,x,a=document.MM_sr; for(i=0;a&&i<a.length&&(x=a[i])&&x.oSrc;i++) x.src=x.oSrc;
}

function MM_openBrWindow(theURL,winName,features) { //v2.0
  window.open(theURL,winName,features);
}

/**
 * Determines if character is numeric.
 *
 * @param lchar Input character to check.
 */
function _isNumeric(lchar)
{
  // Get ASCII code for the particular char
  var cCode = lchar.charCodeAt(0);

  /* ASCII character values:
  48: Number 0.(All values lesser than this is excluded)
  57: Number 9. (All values greater than this is excluded)
  */
  if ((cCode<48)||(cCode>57))
    return false;
  else
    return true;
}

/**
 * Check numeric input of text box(es) for Protocol Term or Age menu.
 *
 * @param event Event Handler.
 * @param arguments processRange method arguments.
 */
function _checkNumericInput(event, id)
{
   var userInput = document.getElementById(id);

   // Get the length of the string
   var strLength = userInput.value.length;

   // Keep track of the decimal count
   var decimalCount = 0;

   // 1) To handle cases where user types input
   if(event.type=='keyup' || event.type=='keypress'){

    // Do not allow user to enter more than one decimal point
    for (i=0; i<strLength; i++){
      if (userInput.value.charAt(i)== '.'){decimalCount++;}
    }
			
    // To get the ASCII code for each character
    var lchar = userInput.value.charAt(strLength-1);
	
    // If char is not numeric and contain more than 1 decimal point
    if ((!_isNumeric(lchar)) ||(decimalCount>1)){
	
      // Remove the illegal characters from the text input
      var number = userInput.value.substring(0,strLength-1);
      userInput.value = number;
    }
  }
  // 2) To handle cases where the user copy-paste input
  else {
    // Keep track of the valid input chars
    var finalIndex = 0;

    // Loop through every character in the string
    for (j=0; j<strLength; j++){

      // Count decimal points
      if (userInput.value.charAt(j)== '.'){decimalCount++;}

      if (!_isNumeric(userInput.value.charAt(j)) || decimalCount >1) {
        finalIndex = j;
        break;
      }
    }
    var number = userInput.value.substring(0,j);
    if (number=='.') { number = ''; }
       userInput.value = number;
  }
  return true;
}

function checkInput(id, arg)
{
  var element = document.getElementById(id);
  YAHOO.util.Event.addListener(element, 'keyup', _checkNumericInput, arg);
  YAHOO.util.Event.addListener(element, 'keypress', _checkNumericInput, arg);
  YAHOO.util.Event.addListener(element, 'blur', _checkNumericInput, arg);
}
//-->
</script>


<title>LONI |  | Data Use Application | Summary</title>
</HEAD>



<BODY leftmargin="0" topmargin="0"  marginwidth="0" marginheight="0" >

 
<title>LONI: Laboratory of Neuro Imaging</title>


<div align="center">  
  <table width="780" border="0" cellpadding="0" cellspacing="0" id="mainContentArea">
   <tr>
    <td align="center" valign="top">
     <table width="780" border="0" cellpadding="0" cellspacing="0" id="standardContent">
      <tr>
       <td width="100%" valign="top"><table width="100%" border="0" cellspacing="0" cellpadding="0">
        <tr>
         <td><a name="Top"></a></td>
        </tr>
        <tr>
         <td height="20"></td>
        </tr>
        <tr align="left" valign="top">
         <td width="100%" height="100">
          
          <table width="100%" border="0" cellspacing="0" cellpadding="0">
           <tr>
            <td>
             <table width="100%" border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td width="15" bgcolor="#666666"></td>
               <td width="7"></td>
               <td  align="left" valign="top"><table width="100%"border="0" cellpadding="0" cellspacing="0">
                <tr>
                 <td height="20" class="Breadcrumb">
                  
                 <a href=http://www.loni.usc.edu>LONI</a> &gt;
                  <a href=></a>
                  
                 </td>
                </tr>
                <tr>
                 <td valign="bottom">
                  
                  <h2>Not Authorized </h2>
                  
                 </td>
                </tr>
               </table>
              </td>
             </tr>
            </table>
           </td>
          </tr>
          <tr>
           <td ><img src="/collaboration/images/PageDiv_Black_750.gif" width="750" height="2"></td>
          </tr>
          <tr>
           <td height="10" ></td>
          </tr>
         </table>
         
        </td>
       </tr>
       <tr align="left" valign="top">

       
        <td>
         <span class="bodyTextAlert">You are not authorized to view or edit this account.
         </span>
        </td>
     
              
     
    </tr>
    <tr>
      <td height="20">
       <img src="/collaboration/images/Spacer_white.gif"  width="560" height="20">
      </td>
    </tr>
   </table></td></tr>
  </table></td></tr>
 </table>
</div>



 <!--
COPYRIGHT NOTICE
Copyright (c) 2018  Scott C. Neu and Arthur W. Toga

See README.license for license notices.
-->




























<div class="footer">
    <div class="footer-inner"> 
      <div class="footer-loni-logo" style="background-image: url('https://ida.loni.usc.edu/services//Images/home//LONI_USC_logo.png')">
        <a href="http://www.loni.usc.edu/" target=_blank ></a>
      </div>
      
      <div class="footer-copyright">
          &#169; 2003 - 2022 LONI. All rights reserved.
      </div>
      
      <div class="footer-usc-logo" style="background-image: url('https://ida.loni.usc.edu/services//Images/home//usc-sublogo2.png')">
        <a href="http://www.ini.usc.edu" target=_blank ></a>
      </div>
    </div>  
</div>














</BODY>



</HTML>
